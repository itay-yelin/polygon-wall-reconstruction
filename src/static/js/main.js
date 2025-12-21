class App {
    constructor() {
        this.inputCanvas = document.getElementById('inputCanvas');
        this.outputCanvas = document.getElementById('outputCanvas');
        this.inputCtx = this.inputCanvas.getContext('2d');
        this.outputCtx = this.outputCanvas.getContext('2d');

        this.lines = [];
        this.polygons = [];

        // Interaction State
        this.isDraggingFn = false;
        this.draggedPoints = []; // [NEW] Array of { lineIndex, pointType: 'start'|'end' }
        this.draggedPoint = null; // Left for compatibility/highlighting single
        this.hoveredPoint = null;
        this.hoveredLineIndex = -1;
        this.hoveredPolygonIndex = -1;

        this.pointRadius = 6;

        // View Transform
        this.scale = 1;
        this.offset = { x: 0, y: 0 };
        this.isPanning = false;
        this.panStart = { x: 0, y: 0 };

        this.init();
    }

    init() {
        this.resizeCanvases();
        window.addEventListener('resize', () => this.resizeCanvases());

        // Controls
        document.getElementById('fileInput').addEventListener('change', (e) => this.handleFileLoad(e));
        document.getElementById('detectBtn').addEventListener('click', () => this.detectPolygons());

        const sampleSelect = document.getElementById('sampleSelect');
        if (sampleSelect) {
            sampleSelect.addEventListener('change', (e) => this.loadSample(e.target.value));

            // [NEW] Checkbox listener
            document.getElementById('showMissing').addEventListener('change', () => this.draw());

            // Initial setup
            this.loadSampleList();
        }

        // Input Canvas Interaction (Edit + View)
        this.setupCanvasEvents(this.inputCanvas, true);
        // Output Canvas Interaction (View Only)
        this.setupCanvasEvents(this.outputCanvas, false);

        this.drawLoop();
    }

    async loadSampleList() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.status === 'success') {
                const sel = document.getElementById('sampleSelect');
                data.files.forEach(f => {
                    const opt = document.createElement('option');
                    opt.value = f;
                    opt.textContent = f;
                    sel.appendChild(opt);
                });
            }
        } catch (e) { console.error(e); }
    }

    async loadSample(filename) {
        if (!filename) return;
        try {
            const res = await fetch(`/api/sample/${filename}`);
            const data = await res.json();
            this.parseLines(data);
            this.polygons = [];
            this.autoScale();
            this.draw();
            document.getElementById('statsContainer').style.display = 'none'; // [NEW] Reset stats
            this.detectPolygons(); // Auto-detect
        } catch (e) { console.error(e); }
    }

    resizeCanvases() {
        const resetCanvas = (canvas, container) => {
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        };
        resetCanvas(this.inputCanvas, this.inputCanvas.parentElement);
        resetCanvas(this.outputCanvas, this.outputCanvas.parentElement);
        this.draw();
    }

    handleFileLoad(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                this.parseLines(data);
                this.autoScale();
                this.polygons = [];
                this.draw();
                document.getElementById('statsContainer').style.display = 'none'; // [NEW] Reset stats
                this.detectPolygons(); // Auto-detect
            } catch (err) {
                alert("Error parsing JSON: " + err.message);
            }
        };
        reader.readAsText(file);
    }

    parseLines(data) {
        this.lines = [];
        const entities = data.entities || [];
        entities.forEach(e => {
            if (e.type === 'line' || (!e.type && e.start_point)) {
                this.lines.push({
                    start: { x: parseFloat(e.start_point.x), y: parseFloat(e.start_point.y) },
                    end: { x: parseFloat(e.end_point.x), y: parseFloat(e.end_point.y) }
                });
            }
        });
    }

    autoScale() {
        if (this.lines.length === 0) return;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        this.lines.forEach(l => {
            minX = Math.min(minX, l.start.x, l.end.x);
            minY = Math.min(minY, l.start.y, l.end.y);
            maxX = Math.max(maxX, l.start.x, l.end.x);
            maxY = Math.max(maxY, l.start.y, l.end.y);
        });

        const padding = 40;
        const width = maxX - minX || 100;
        const height = maxY - minY || 100;

        const canvasW = this.inputCanvas.width;
        const canvasH = this.inputCanvas.height;

        const scaleX = (canvasW - padding * 2) / width;
        const scaleY = (canvasH - padding * 2) / height;

        this.scale = Math.min(scaleX, scaleY);

        // Center it (reset pan)
        this.offset.x = padding - minX * this.scale + (canvasW - padding * 2 - width * this.scale) / 2;
        this.offset.y = padding - minY * this.scale + (canvasH - padding * 2 - height * this.scale) / 2;
    }

    worldToScreen(x, y) {
        return {
            x: x * this.scale + this.offset.x,
            y: y * this.scale + this.offset.y
        };
    }

    screenToWorld(x, y) {
        return {
            x: (x - this.offset.x) / this.scale,
            y: (y - this.offset.y) / this.scale
        };
    }

    getMousePos(evt, canvas) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: evt.clientX - rect.left,
            y: evt.clientY - rect.top
        };
    }

    dist(p1, p2) {
        return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
    }

    distToSegment(p, a, b) {
        const l2 = (b.x - a.x) ** 2 + (b.y - a.y) ** 2;
        if (l2 === 0) return this.dist(p, a);
        let t = ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2;
        t = Math.max(0, Math.min(1, t));
        const proj = { x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) };
        return this.dist(p, proj);
    }

    isPointInPolygon(p, polyPoints) {
        let oddNodes = false;
        let j = polyPoints.length - 1;
        for (let i = 0; i < polyPoints.length; i++) {
            const xi = polyPoints[i][0], yi = polyPoints[i][1];
            const xj = polyPoints[j][0], yj = polyPoints[j][1];
            if ((yi < p.y && yj >= p.y || yj < p.y && yi >= p.y) &&
                (xi <= p.x || xj <= p.x)) {
                if (xi + (p.y - yi) / (yj - yi) * (xj - xi) < p.x) {
                    oddNodes = !oddNodes;
                }
            }
            j = i;
        }
        return oddNodes;
    }

    findPointUnderMouse(mousePos) {
        const threshold = 10;
        for (let i = 0; i < this.lines.length; i++) {
            const line = this.lines[i];
            const s = this.worldToScreen(line.start.x, line.start.y);
            const e = this.worldToScreen(line.end.x, line.end.y);

            if (this.dist(mousePos, s) < threshold) return { index: i, type: 'start' };
            if (this.dist(mousePos, e) < threshold) return { index: i, type: 'end' };
        }
        return null;
    }

    // [NEW] Find all points coincident (or very close) to a given point
    findCoincidentPoints(point) {
        const threshold = 1; // Since we are comparing World coordinates (if exact match)
        // Actually, let's use the mouse hit test threshold logic but in screen space or 
        // check precise equality in world space if they are snapped.
        // Let's assume they are snapped.

        const targetLine = this.lines[point.index];
        const targetPt = point.type === 'start' ? targetLine.start : targetLine.end;

        const points = [];
        this.lines.forEach((l, i) => {
            if (this.dist(l.start, targetPt) < 0.001) points.push({ index: i, type: 'start' });
            if (this.dist(l.end, targetPt) < 0.001) points.push({ index: i, type: 'end' });
        });
        return points;
    }

    findLineUnderMouse(mousePos) {
        const threshold = 5;
        for (let i = 0; i < this.lines.length; i++) {
            const line = this.lines[i];
            const s = this.worldToScreen(line.start.x, line.start.y);
            const e = this.worldToScreen(line.end.x, line.end.y);

            if (this.distToSegment(mousePos, s, e) < threshold) {
                return i;
            }
        }
        return -1;
    }

    findPolygonUnderMouse(mousePos) {
        const worldPos = this.screenToWorld(mousePos.x, mousePos.y);
        for (let i = 0; i < this.polygons.length; i++) {
            const poly = this.polygons[i];
            if (!poly.points) continue;
            if (this.isPointInPolygon(worldPos, poly.points)) return i;
        }
        return -1;
    }

    setupCanvasEvents(canvas, isEditable) {
        const isInput = (canvas === this.inputCanvas);

        canvas.addEventListener('mousedown', (e) => {
            const mouse = this.getMousePos(e, canvas);

            if (isEditable) {
                const point = this.findPointUnderMouse(mouse);
                if (point && e.button === 0) {
                    this.isDraggingFn = true;
                    this.draggedPoint = point; // For highlighting primary
                    this.draggedPoints = this.findCoincidentPoints(point); // [NEW] grab all attached
                    return;
                }
            }

            this.isPanning = true;
            this.panStart = { x: mouse.x - this.offset.x, y: mouse.y - this.offset.y };
            canvas.style.cursor = 'grabbing';
        });

        canvas.addEventListener('mousemove', (e) => {
            const mouse = this.getMousePos(e, canvas);

            if (this.isDraggingFn && this.draggedPoints.length > 0 && isEditable) {
                const worldPos = this.screenToWorld(mouse.x, mouse.y);

                // [NEW] Update ALL dragged points
                this.draggedPoints.forEach(dp => {
                    const line = this.lines[dp.index];
                    if (dp.type === 'start') {
                        line.start.x = worldPos.x;
                        line.start.y = worldPos.y;
                    } else {
                        line.end.x = worldPos.x;
                        line.end.y = worldPos.y;
                    }
                });

            } else if (this.isPanning) {
                this.offset.x = mouse.x - this.panStart.x;
                this.offset.y = mouse.y - this.panStart.y;
            } else {
                if (isEditable) {
                    const hit = this.findPointUnderMouse(mouse);
                    canvas.style.cursor = hit ? 'move' : 'grab';
                    this.hoveredPoint = hit;
                    this.hoveredLineIndex = this.findLineUnderMouse(mouse);
                } else {
                    canvas.style.cursor = 'grab';
                    this.hoveredPolygonIndex = this.findPolygonUnderMouse(mouse);
                }
            }
            this.draw();
        });

        const endInteraction = () => {
            this.isDraggingFn = false;
            this.draggedPoint = null;
            this.draggedPoints = [];
            this.isPanning = false;
            canvas.style.cursor = 'default';
        };

        canvas.addEventListener('mouseup', endInteraction);
        canvas.addEventListener('mouseleave', endInteraction);

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const mouse = this.getMousePos(e, canvas);
            const worldBefore = this.screenToWorld(mouse.x, mouse.y);

            const zoomFactor = 1.1;
            if (e.deltaY < 0) {
                this.scale *= zoomFactor;
            } else {
                this.scale /= zoomFactor;
            }

            this.offset.x = mouse.x - worldBefore.x * this.scale;
            this.offset.y = mouse.y - worldBefore.y * this.scale;

            this.draw();
        });
    }

    async detectPolygons() {
        if (this.lines.length === 0) return;

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lines: this.lines })
            });
            const result = await response.json();

            if (result.status === 'success') {
                this.polygons = result.polygons;
                this.draw();

                // [NEW] Show Stats
                const statsDiv = document.getElementById('statsContainer');
                const valSpan = document.getElementById('coverageVal');

                // Reset missing lines
                this.missingLines = [];

                if (result.stats) {
                    if (result.stats.coverage_pct !== undefined) {
                        statsDiv.style.display = 'inline-flex';
                        valSpan.textContent = result.stats.coverage_pct.toFixed(1) + '%';
                    }
                    if (result.stats.missing_lines) {
                        this.missingLines = result.stats.missing_lines;
                    }
                }
                // Redraw to show missing lines if checked
                this.draw();
            } else {
                alert("Detection failed: " + result.message);
            }
        } catch (err) {
            alert("Error connecting to server: " + err.message);
        }
    }

    draw() {
        this.drawInput();
        this.drawOutput();
    }

    drawInput() {
        const ctx = this.inputCtx;
        ctx.clearRect(0, 0, this.inputCanvas.width, this.inputCanvas.height);

        ctx.save();

        this.lines.forEach((line, i) => {
            const s = this.worldToScreen(line.start.x, line.start.y);
            const e = this.worldToScreen(line.end.x, line.end.y);

            ctx.beginPath();
            ctx.moveTo(s.x, s.y);
            ctx.lineTo(e.x, e.y);

            if (i === this.hoveredLineIndex) {
                ctx.lineWidth = 5;
                ctx.strokeStyle = '#d32f2f';
            } else {
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#333';
            }
            ctx.stroke();

            // Draw points
            const drawPt = (pt) => {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#ff4444';
                ctx.fill();
            };
            drawPt(s);
            drawPt(e);
        });

        // Highlight dragged point
        const active = this.draggedPoint || this.hoveredPoint;
        if (active) {
            const line = this.lines[active.index];
            const pt = active.type === 'start' ? line.start : line.end;
            const s = this.worldToScreen(pt.x, pt.y);

            ctx.beginPath();
            ctx.arc(s.x, s.y, 8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 200, 0, 0.5)';
            ctx.fill();
            ctx.stroke();
        }
        ctx.restore();
    }

    drawOutput() {
        const ctx = this.outputCtx;
        ctx.clearRect(0, 0, this.outputCanvas.width, this.outputCanvas.height);

        // Draw standard polygons
        // Grid drawing if needed (omitted for now as clearly removed in previous broken edit)

        ctx.lineWidth = 2;
        this.polygons.forEach((poly, i) => {
            const isHovered = (i === this.hoveredPolygonIndex);

            ctx.strokeStyle = isHovered ? '#0043ce' : '#0d6efd';
            ctx.fillStyle = isHovered ? 'rgba(13, 110, 253, 0.3)' : 'rgba(13, 110, 253, 0.1)';

            ctx.beginPath();
            poly.points.forEach((p, idx) => {
                const s = this.worldToScreen(p[0], p[1]);
                if (idx === 0) ctx.moveTo(s.x, s.y);
                else ctx.lineTo(s.x, s.y);
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        });

        // Draw Missing Lines
        const showMissing = document.getElementById('showMissing');
        if (showMissing && showMissing.checked && this.missingLines && this.missingLines.length > 0) {
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 1;
            ctx.beginPath();
            this.missingLines.forEach(lineSeg => {
                if (lineSeg.length < 2) return;
                const s1 = this.worldToScreen(lineSeg[0][0], lineSeg[0][1]);
                ctx.moveTo(s1.x, s1.y);
                for (let k = 1; k < lineSeg.length; k++) {
                    const sk = this.worldToScreen(lineSeg[k][0], lineSeg[k][1]);
                    ctx.lineTo(sk.x, sk.y);
                }
            });
            ctx.stroke();
        }
    }

    drawLoop() {
        requestAnimationFrame(() => this.drawLoop());
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new App();
});

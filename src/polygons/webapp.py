from flask import Flask, render_template, request, jsonify
import sys
import os

from .detect import detect_polygons
from .io_json import Line, Point

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        lines_data = data.get('lines', [])
        
        # Convert dictionary data to Line objects
        input_lines = []
        for l in lines_data:
            start = Point(float(l['start']['x']), float(l['start']['y']))
            end = Point(float(l['end']['x']), float(l['end']['y']))
            input_lines.append(Line(start, end))

        # Run detection
        import time
        t0 = time.time()
        polygons, stats = detect_polygons(input_lines)
        dt = time.time() - t0
        
        # Add time to stats
        stats['time_sec'] = dt

        return jsonify({
            'status': 'success', 
            'polygons': polygons,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/samples', methods=['GET'])
def list_samples():
    # Data is expected to be in the project root/data
    # We assume CWD is project root or we find it relative to this file
    # ../../../data would be from src/polygons/webapp.py -> src/polygons -> src -> root -> data
    # But installed package might be different. Let's assume running from root or data is distributed.
    # For this assignment, we assume running from repo root.
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
         # Try relative check for dev mode
         data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    
    try:
        if not os.path.exists(data_dir):
            return jsonify({'status': 'success', 'files': []})
            
        files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        return jsonify({'status': 'success', 'files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/sample/<filename>', methods=['GET'])
def get_sample(filename):
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
         data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
         
    filepath = os.path.join(data_dir, filename)
    try:
        if not os.path.exists(filepath):
             return jsonify({'status': 'error', 'message': 'File not found'}), 404
        with open(filepath, 'r') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def main():
    # Allow running with python -m polygons.webapp
    debug = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    app.run(host='127.0.0.1', port=5000, debug=debug)

if __name__ == '__main__':
    main()

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ABRSM_MODEL_PATH       = os.path.join(BASE_DIR, 'models', 'model_abrsm_ujian.pkl')
COMPETITION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_competition.pkl')

COMPETITION_FEATURE_COLS = [
    'tempo_control',
    'accuracy_cleanliness',
    'hand_coordination',
    'dynamics_articulation',
    'expression_emotion',
    'phrasing',
    'stage_presence'
]

# ── Load ABRSM model ──────────────────────────────────────────────────────────

try:
    abrsm_package              = joblib.load(ABRSM_MODEL_PATH)
    abrsm_model                = abrsm_package['model']
    abrsm_feature_names        = abrsm_package['feature_names']
    abrsm_reverse_label        = abrsm_package['reverse_label_mapping']
    abrsm_metadata             = abrsm_package['metadata']
    ABRSM_LOADED               = True
    print(f"ABRSM model loaded — accuracy: {abrsm_metadata.get('test_accuracy', 'N/A')}")
except Exception as e:
    ABRSM_LOADED = False
    print(f"ABRSM model failed to load: {e}")

# ── Load Competition model ────────────────────────────────────────────────────

try:
    comp_package               = joblib.load(COMPETITION_MODEL_PATH)
    comp_model                 = comp_package['model']
    comp_feature_names         = comp_package['feature_names']
    comp_label_mapping         = comp_package['label_mapping']
    comp_reverse_label         = comp_package['reverse_label_mapping']
    comp_metadata              = comp_package['metadata']
    COMPETITION_LOADED         = True
    print(f"Competition model loaded — accuracy: {comp_metadata.get('test_accuracy', 'N/A')}")
except Exception as e:
    COMPETITION_LOADED = False
    print(f"Competition model failed to load: {e}")


# ── Health check ──────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'success':   True,
        'status':    'healthy',
        'models': {
            'abrsm':       ABRSM_LOADED,
            'competition': COMPETITION_LOADED
        },
        'timestamp': datetime.now().isoformat()
    }), 200


# ── ABRSM endpoints ───────────────────────────────────────────────────────────

@app.route('/abrsm/predict', methods=['POST'])
def abrsm_predict():
    if not ABRSM_LOADED:
        return jsonify({'success': False, 'message': 'Model ABRSM belum dimuat.'}), 500

    try:
        data = request.json

        for field in ['murid_id', 'goal_id', 'grade', 'nilai']:
            if field not in data:
                return jsonify({'success': False, 'message': f'Field wajib tidak ada: {field}'}), 400

        nilai = data['nilai']
        grade = int(data['grade'])

        for cat in ['lagu', 'scales', 'sight', 'aural']:
            if cat not in nilai:
                return jsonify({'success': False, 'message': f'Kategori nilai tidak ada: {cat}'}), 400

        validations = [
            ('lagu',   nilai['lagu'],   0, 90),
            ('scales', nilai['scales'], 0, 21),
            ('sight',  nilai['sight'],  0, 21),
            ('aural',  nilai['aural'],  0, 18),
        ]
        for name, val, lo, hi in validations:
            v = float(val)
            if v < lo or v > hi:
                return jsonify({'success': False, 'message': f'Nilai {name} harus {lo}-{hi}, diterima: {v}'}), 400

        if grade < 1 or grade > 8:
            return jsonify({'success': False, 'message': 'Grade harus 1-8'}), 400

        lagu_avg = float(nilai['lagu']) / 3.0
        features = np.array([[grade, lagu_avg, float(nilai['scales']), float(nilai['sight']), float(nilai['aural'])]])

        prediction    = abrsm_model.predict(features)[0]
        probabilities = abrsm_model.predict_proba(features)[0]
        result        = abrsm_reverse_label[prediction]
        confidence    = float(probabilities[prediction])
        total_score   = float(nilai['lagu']) + float(nilai['scales']) + float(nilai['sight']) + float(nilai['aural'])

        return jsonify({
            'success': True,
            'prediction': {
                'result':                result,
                'confidence':            round(confidence, 4),
                'confidence_percentage': round(confidence * 100, 2),
                'probabilities': {
                    'Fail':        round(float(probabilities[0]), 4),
                    'Pass':        round(float(probabilities[1]), 4),
                    'Merit':       round(float(probabilities[2]), 4),
                    'Distinction': round(float(probabilities[3]), 4)
                }
            },
            'input_data': {
                'murid_id':    data['murid_id'],
                'goal_id':     data['goal_id'],
                'grade':       grade,
                'nilai':       nilai,
                'lagu_avg':    round(lagu_avg, 2),
                'total_score': round(total_score, 2)
            },
            'model_info':  {'training_date': abrsm_package.get('training_date', 'Unknown')},
            'timestamp':   datetime.now().isoformat()
        }), 200

    except ValueError as e:
        return jsonify({'success': False, 'message': f'Nilai tidak valid: {str(e)}'}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': f'Prediction error: {str(e)}'}), 500


@app.route('/abrsm/health', methods=['GET'])
def abrsm_health():
    return jsonify({'success': True, 'model_loaded': ABRSM_LOADED, 'timestamp': datetime.now().isoformat()}), 200


@app.route('/abrsm/model-info', methods=['GET'])
def abrsm_model_info():
    if not ABRSM_LOADED:
        return jsonify({'success': False, 'message': 'Model belum dimuat'}), 500
    return jsonify({
        'success': True,
        'model_info': {
            'features':      abrsm_feature_names,
            'labels':        list(abrsm_reverse_label.values()),
            'test_accuracy': abrsm_metadata.get('test_accuracy', 'N/A'),
            'cv_score':      abrsm_metadata.get('cv_mean', 'N/A'),
            'training_date': abrsm_package.get('training_date', 'Unknown'),
            'n_samples':     abrsm_metadata.get('n_samples', 'N/A')
        }
    }), 200


# ── Competition endpoints ─────────────────────────────────────────────────────

@app.route('/competition/predict', methods=['POST'])
def competition_predict():
    if not COMPETITION_LOADED:
        return jsonify({'success': False, 'message': 'Model Competition belum dimuat.'}), 500

    try:
        data = request.json

        for field in ['murid_id', 'goal_id', 'nilai']:
            if field not in data:
                return jsonify({'success': False, 'message': f'Field wajib tidak ada: {field}'}), 400

        nilai = data['nilai']

        for rubrik in COMPETITION_FEATURE_COLS:
            if rubrik not in nilai:
                return jsonify({'success': False, 'message': f'Rubrik tidak ada: {rubrik}'}), 400
            try:
                v = float(nilai[rubrik])
            except (TypeError, ValueError):
                return jsonify({'success': False, 'message': f'Nilai {rubrik} harus angka'}), 400
            if v < 0.0 or v > 3.0:
                return jsonify({'success': False, 'message': f'Nilai {rubrik} harus 0.0-3.0, diterima: {v}'}), 400

        features = np.array([[
            float(nilai['tempo_control']),
            float(nilai['accuracy_cleanliness']),
            float(nilai['hand_coordination']),
            float(nilai['dynamics_articulation']),
            float(nilai['expression_emotion']),
            float(nilai['phrasing']),
            float(nilai['stage_presence'])
        ]])

        prediction    = comp_model.predict(features)[0]
        probabilities = comp_model.predict_proba(features)[0]
        result        = comp_reverse_label[prediction]
        confidence    = float(probabilities[prediction])

        return jsonify({
            'success': True,
            'prediction': {
                'result':                result,
                'confidence':            round(confidence, 4),
                'confidence_percentage': round(confidence * 100, 2),
                'probabilities': {
                    'Not Ready':   round(float(probabilities[0]), 4),
                    'Developing':  round(float(probabilities[1]), 4),
                    'Ready':       round(float(probabilities[2]), 4),
                    'Competitive': round(float(probabilities[3]), 4)
                }
            },
            'input_data': {
                'murid_id': data['murid_id'],
                'goal_id':  data['goal_id'],
                'nilai':    {k: float(v) for k, v in nilai.items() if k in COMPETITION_FEATURE_COLS}
            },
            'model_info':  {'training_date': comp_package.get('training_date', 'Unknown')},
            'timestamp':   datetime.now().isoformat()
        }), 200

    except ValueError as e:
        return jsonify({'success': False, 'message': f'Nilai tidak valid: {str(e)}'}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': f'Prediction error: {str(e)}'}), 500


@app.route('/competition/health', methods=['GET'])
def competition_health():
    return jsonify({'success': True, 'model_loaded': COMPETITION_LOADED, 'timestamp': datetime.now().isoformat()}), 200


@app.route('/competition/model-info', methods=['GET'])
def competition_model_info():
    if not COMPETITION_LOADED:
        return jsonify({'success': False, 'message': 'Model belum dimuat'}), 500
    return jsonify({
        'success': True,
        'model_info': {
            'features':      comp_feature_names,
            'labels':        ['Not Ready', 'Developing', 'Ready', 'Competitive'],
            'test_accuracy': comp_metadata.get('test_accuracy', 'N/A'),
            'cv_score':      comp_metadata.get('cv_mean', 'N/A'),
            'training_date': comp_package.get('training_date', 'Unknown'),
            'n_samples':     comp_metadata.get('n_samples', 'N/A')
        }
    }), 200


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

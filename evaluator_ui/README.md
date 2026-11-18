# PathSnap Laptop 2 Evaluator Client

This lightweight browser client sends GPS and road-graph input to the FastAPI server on Laptop 1. It contains no route labels or evaluator ground truth.

## Laptop 1

From the project root, start the API using the tuned model:

```bash
PATHSNAP_MODEL=artifacts/models/histgb-20261118T070402Z.pkl uvicorn app:app --host 0.0.0.0 --port 8000
```

The default settings are 24 road candidates and the cross-validation-selected Viterbi transition weight of 0. Confirm the service at `http://127.0.0.1:8000/health`.

## Laptop 2

From the project root, serve the client files on the shared network:

```bash
python -m http.server 8080 --bind 0.0.0.0 --directory evaluator_ui
```

Open `http://127.0.0.1:8080` on Laptop 2. In the API base URL field, enter `http://<LAPTOP_1_IP>:8000`, paste the evaluator's track/nodes/arcs request JSON, and click **Predict route**. Both laptops must be on a network that allows connections between them.

The API enables CORS for the browser demo. Set `PATHSNAP_CORS_ORIGINS` to a comma-separated origin allowlist when the evaluator page has a known origin.

## Limits

The event's actual evaluator interface and its request schema were not present in the workspace, and there is no evaluator-test partition. This client is a compatible demonstration form for the project's `/predict` contract; adapt the input mapping if the organizer supplies a different interface contract.

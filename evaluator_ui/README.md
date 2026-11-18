# PathSnap evaluator UI

`index.html` is the runnable Laptop 2 client. It supports evaluation trips listed by the API and uploaded trip JSON or `.track`, `.nodes`, and `.arcs` files. It never receives or displays reference `.route` labels.

## Laptop 1: API

From the repository root, use the newly fitted 17-feature model:

```bash
PATHSNAP_DATA=. PATHSNAP_EVALUATOR_SPLIT=evaluator_test PATHSNAP_MODEL=artifacts/models/histgb-context17-20261118T103543Z.pkl uvicorn app:app --host 0.0.0.0 --port 8000
```

The API lists trips from the local `evaluator_test/` directory. It opens only `.track`, `.nodes`, and `.arcs`; it never loads `.route`. The model sidecar configures 40 candidates and context radius 2. Check `http://127.0.0.1:8000/health` on Laptop 1. Set `PATHSNAP_CORS_ORIGINS` to the UI origin if you do not want the default wildcard CORS policy.

## Laptop 2: UI

From the repository root, serve the page on the shared network:

```bash
python3 -m http.server 8080 --bind 0.0.0.0 --directory evaluator_ui
```

Open `http://127.0.0.1:8080` on Laptop 2. Set **Laptop 1 API address** to `http://<LAPTOP_1_IP>:8000`, click **Check connection**, then choose an evaluation trip or upload a trip. Both laptops must be on a network that permits connections between them.

The API reads dataset trips without loading their `.route` file. Both dataset and upload map endpoints render only GPS, the road network, matched positions, and the predicted route.

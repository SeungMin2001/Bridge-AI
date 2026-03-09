# Run Guide

### 1. Clone Repository

```bash
git clone https://github.com/SeungMin2001/Group-Chat-agent
cd Group-Chat-agent
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Download Model (First Run Only)

```bash
python scripts/download_model.py
```

This downloads the model to your local machine.

---

### 4. Run the Agents

```bash
python app/main.py
```

---

### Notes

* The model is downloaded **only once**.
* After the first download, running `python app/main.py` will reuse the existing model.
* Internet is required only for the first model download.

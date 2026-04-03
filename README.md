# 🏨 Hotel Demand Intelligence Dashboard

A production-ready Streamlit web app converting the Hotel Demand Intelligence Jupyter Notebook into an interactive dashboard.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ✨ Features

| Section | Description |
|---|---|
| **KPI Cards** | Total bookings, cancellation rate, ADR, avg stay, repeat guests |
| **Booking Trends** | Monthly bookings, lead time distribution, customer type |
| **Cancellation Analysis** | By segment, deposit type, special requests, hotel type, month |
| **Revenue & ADR** | Monthly ADR trends, revenue area charts, country heatmap |
| **Prediction** | Real-time cancellation risk with gauge + probability scores |
| **Insights** | 6 actionable business recommendations |

## 🔧 Sidebar Filters
- Hotel Type (City / Resort)
- Market Segment (multiselect)
- Room Type (multiselect)
- Lead Time slider
- Booking Status (All / Confirmed / Cancelled)

---

## 📦 Dependencies

- `streamlit` — web app framework
- `pandas` — data manipulation
- `numpy` — numerical computing
- `plotly` — interactive charts
- `scikit-learn` — ML model (Gradient Boosting)

> **Note:** The dataset is loaded directly from a public GitHub URL — no local file needed.

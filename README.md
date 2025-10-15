AI Data Agent  

A lightweight **Streamlit application** that performs quick exploratory data analysis, anomaly detection, and generates short written insights using an open‑source language model (Microsoft Phi‑2).

(1) Why this project
	
	Collecting and cleaning data always takes longer than explaining what it means.  
This small tool automates the first half of that cycle:
	* accepts a CSV / Excel file,  
* creates a detailed EDA report,  
* surfaces outliers automatically, and  
* writes a few clear business recommendations.
	It’s meant as a reusable framework that analysts can extend with their own models or datasets.

(2) How it works
	
	1. **Upload data** through the browser (CSV or XLSX).  
2. **EDA Report** – ydata‑profiling builds an interactive HTML summary with distributions, correlations, and missing‑value checks.  
3. **Anomaly Detection** – Isolation Forest + Z‑score identify unusual numerical values.  
4. **AI Insights** – statistical summaries are sent to the small open model (Phi‑2) to generate 3–5 recommendations on data quality, new analytics angles, and dashboard ideas.  
5. **Download** – results are displayed and saved as `ai_insights.txt`.


(3) Tech stack

	| Purpose | Library / Tool |
|----------|----------------|
| Interface | Streamlit |
| EDA profiling | ydata‑profiling (formerly pandas‑profiling) |
| Anomalies | scikit‑learn (Isolation Forest), scipy (Z‑score) |
| Data handling | pandas, numpy |
| Language model | transformers + torch (Microsoft Phi‑2) |
| Optimisation | accelerate, bitsandbytes |

(4) File structure

	AI_Data_Agent/
	│
	├── ai_agent.py # reusable analysis utilities
	├── ai_agent_app.py # Streamlit web app entry point
	├── generate_data.py # creates demo business_data.csv
	├── requirements.txt # required libraries
	├── README.md # this file
	└── Documentation.txt # detailed documentation

(5) ## 🧩 Setup Bash
# create environment

	python -m venv venv
venv\Scripts\activate               # on Windows
	
	
	# install dependencies
          pip install -r requirements.txt
	# run the application
        streamlit run ai_agent_app.py

(6) 🧪 Try it quickly

	Generate a synthetic dataset using generate.py:
	
	Bash
	python generate_data.py
	Then open http://localhost:8501, upload business_data.csv, and explore the outputs:
	• EDA Report
	• Anomaly Detection tables
	• AI‑Generated Insights file

(7) Example insight

	• Online channel yields the highest margins; highlight it in future dashboards.  
• Missing or abnormally low unit prices denote manual entry errors.  
• Add time‑series visuals to reveal monthly sales peaks.  
• Automate report refresh with an ETL pipeline and scheduled Streamlit runs.

(8) Future work

	• Model selector (Gemma, Mistral, Phi‑2).
	• Power BI / Tableau export.
	• Database connections (MySQL / Snowflake).
	• Forecasting module or textual Q&A on EDA results.

(9) Author :
	Hardik Singh

Project built for learning and demonstrating data‑analysis automation and local LLM integration.

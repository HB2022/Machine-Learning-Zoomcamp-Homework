import requests

url = 'http://0.0.0.0:9696/predict'

customer = {
  "gender": "male",
  "seniorcitizen": 0,
  "partner": "no",
  "dependents": "yes",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 6,
  "monthlycharges": 29.85,
  "totalcharges": 129.85
}

response = requests.post(url, json=customer)

print("Status:", response.status_code)
print("Response:", response.text)  # FastAPI usually names the invalid field here

if response.ok:
    result = response.json()
    print("Response data:", result)
    if result["churn"]:
        print("customer is likely to churn, send promo")
    else:
        print("customer is not likely to churn")
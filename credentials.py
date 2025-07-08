import os, base64

def decode_google_keys():
    alert_b64 = os.getenv("ALERT_JSON_B64")
    if alert_b64:
        with open("alert-parsec.json", "wb") as f:
            f.write(base64.b64decode(alert_b64))

    even_b64 = os.getenv("EVEN_JSON_B64")
    if even_b64:
        with open("even-flight.json", "wb") as f:
            f.write(base64.b64decode(even_b64))

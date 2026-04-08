import smtplib
from email.mime.text import MIMEText

def send_email(to_email, price, city, area):

    sender = "783075kalpna@gmail.com"
    password = "ycqmstydzouqlmyu"

    msg = MIMEText(f"""
House Price Prediction Result

City: {city}
Area: {area}
Predicted Price: ₹{price}
""")

    msg['Subject'] = "House Price Result"
    msg['From'] = sender
    msg['To'] = to_email

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()
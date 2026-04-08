# Smart House Price Prediction System
## Project Overview
This project is a Machine Learning-based web application that predicts house prices based on user inputs such as city, area, square feet, number of bedrooms, and property type. The system provides quick and data-driven results to help users make better property decisions.

---

## Features
- House price prediction using Machine Learning  
- Price range estimation  
- Area score calculation  
- Similar property recommendations  
- Nearby facilities detection  
- Graph visualization  
- Email notification system  

---

## Technologies Used
- Python  
- Flask  
- HTML, CSS, JavaScript  
- Pandas  
- Scikit-learn (Random Forest)  

---

## Machine Learning Model
- Algorithm: Random Forest Regression  
- Type: Supervised Learning  
- Preprocessing: Data cleaning and label encoding  

---

## Project Structure
project/ │ ├── app.py ├── train_model.py ├── model.pkl ├── house_data.csv │ ├── utils/ │   ├── neighborhood.py │   ├── recommendations.py │   ├── facilities.py │   └── email_service.py │ ├── templates/ │   └── index.html │ ├── static/ │   └── style.css │ └── README.md

---

## How to Run
1. Install required libraries:
   pip install flask pandas scikit-learn  

2. Train the model:
   python train_model.py  

3. Run the application:
   python app.py  

4. Open browser:
   http://127.0.0.1:5000/

---

## Working
- User enters input data  
- Data is processed in backend  
- Machine learning model predicts price  
- Additional features are generated  
- Results are displayed on the screen  

---

## Conclusion
This project combines machine learning and web development to create a simple and effective system for house price prediction. It helps users get quick and useful insights for property decisions.

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4bb86859-5bae-4e15-a0a8-891f9ae2736a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-23 12:56:11.536 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Anna Kodji\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import joblib\n",
    "\n",
    "#loading the scaler\n",
    "scaler=joblib.load(r'C:\\Users\\Anna Kodji\\Downloads\\scaler.pkl')\n",
    "\n",
    "\n",
    "#loading the Random Forest Regressor model\n",
    "model=joblib.load(r'C:\\Users\\Anna Kodji\\Downloads\\model.pkl')\n",
    "\n",
    "#loading the list of features used to train the model\n",
    "with open(r'C:\\Users\\Anna Kodji\\Downloads\\top_features_list.pkl','rb') as file:\n",
    "    top_features_list=pickle.load(file)\n",
    "\n",
    "\n",
    "\n",
    "feature_name_mapping ={\n",
    "    \"Value in Euros\":top_features_list[0],\n",
    "    \"Age\":top_features_list[1],\n",
    "    \"Potential\": top_features_list[2],\n",
    "    \"Movement Reactions\": top_features_list[3],\n",
    "    \"Wage in Euros\": top_features_list[4],\n",
    "    \"Mentality Composure\": top_features_list[5],\n",
    "    \"Goalkeeping\": top_features_list[6]\n",
    "}\n",
    "\n",
    "st.title('FIFA Player Rating Predictor')\n",
    "st.sidebar.header('Enter Feature Values')\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "user_inputs={}\n",
    "for user_friendly_name, actual_name in feature_name_mapping.items():\n",
    "    user_inputs[user_friendly_name] = st.sidebar.number_input(f'Enter value for {user_friendly_name}', value=0.0)\n",
    "\n",
    "if st.sidebar.button('Predict'):\n",
    "    input_data=pd.DataFrame({actual_name: user_inputs[user_friendly_name] for user_friendly_name, actual_name in feature_name_mapping.items()}, index=[0])\n",
    "\n",
    "    scaled_input_data=scaler.transform(input_data)\n",
    "\n",
    "    predicted_rating = model.predict(scaled_input_data)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

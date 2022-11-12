import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import streamlit as st

model = tf.keras.models.load_model(
       ("depression_Classifier.h5"),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

def main():
    st.title("Depression prediction")

    #input variables
    message = st.text_input("Write your message")

    #Prediction code
    if st.button("Predict"):
        prediction = model.predict([message])[0][0]
        prediction = prediction*100

        if prediction <= 20:
            st.success("""Your probability of being throught depression is {:.2f}%, it looks like you do not pressent indications
            of it but it would be recomended to seek professional help if you do not feel good""".format(prediction))
        elif prediction<=60:
            st.warning("""Your probability of being throught depression is {:.2f}%, it looks like you present some indications of
            it. We recommend you to seek professional help""".format(prediction))
        else:
            st.error("Your probability of being throught depression is {:.2f}%, we recommend you to seek professional help as soon as possible".format(prediction))

if __name__=='__main__':
    main()
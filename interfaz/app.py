import streamlit as st

st.set_page_config(page_title="App en AWS App Runner")

st.title("¡Hola desde Streamlit en AWS!")
st.write("Esta es una prueba de despliegue en AWS App Runner.")

# Ejecutar Streamlit en el puerto 8080 cuando se despliegue en App Runner
if __name__ == "__main__":
    import os
    os.system("streamlit run app.py --server.port=8080 --server.address=0.0.0.0")


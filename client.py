import requests
import streamlit as st
import base64
import os

st.title("Local ChatBot")

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


all_files = requests.get("http://localhost:8000/upload").json()
# File upload section
st.sidebar.header("File Upload")
uploaded_files = st.sidebar.file_uploader("Choose files to upload", accept_multiple_files=True)
if st.sidebar.button("Upload Files"):
    if uploaded_files:
        files = [("files", file) for file in uploaded_files]
        response = requests.post("http://localhost:8000/upload", files=files)
        if response.status_code == 200:
            st.sidebar.success(f"Successfully uploaded {len(uploaded_files)} file(s)")
            for file in response.json()['files']:
                all_files.append(file)
            if 'skipped_files' in response.json():
                for file in response.json()['skipped_files']:
                    st.sidebar.warning(f"Skipped: {file}")
        else:
            st.sidebar.error("Error uploading files")

st.sidebar.header("Files")
for file in all_files:
    st.sidebar.write(f"- {file}")

top_k = st.sidebar.slider(value= 3, label="Sources to retrieve", min_value=1, max_value=20, step=1)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        st.markdown(content)

if prompt := st.chat_input("Write your prompt in this input field"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"http://localhost:8000/chat", json={"message": prompt, "top_k": top_k}
            ).json()
            final_message = response['response']
            st.markdown(final_message)

            if response['sources'] is not None:
                st.write("Sources:")
                for source in response['sources']:
                    file_path = source['source_path'].split('?')[0]
                    if os.path.exists(file_path):
                            download_link = get_binary_file_downloader_html(file_path, 'PDF')
                            st.markdown(f"- {source['source_name']}, page {source['page']}: {download_link}", unsafe_allow_html=True)
                            final_message += f"\n- {source['source_name']}, page {source['page']}"
                    else:
                        st.markdown(f"- {source['source_name']}: File not found")
            st.session_state.messages.append({"role": "assistant", "content": final_message})

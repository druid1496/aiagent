import ollama
response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'Describe this image in English.',
        'images': ['./lab6/photo.jpg']
    }]
)
print(response['message']['content'])

import openai

openai.api_key = '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'

output = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": 
             "Write me a script for hosting a \
             conference on technology"}]
)

print(output)


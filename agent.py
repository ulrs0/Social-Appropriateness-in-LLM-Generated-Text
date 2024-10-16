from together import Together
from openai import OpenAI

class Agent:
    def __init__(self, model):
        self.client = Together()
        self.model = model
        self.messages = []
    def forward(self, x):
        self.messages.append({"role": "user", "content": x})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=2000,
            temperature=0.7,
        )
        response = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response
    def get_messages(self):
        return self.messages
    def set_system_prompt(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]
    def clear_messages(self):
        self.messages = []

class AgentGPT:
    def __init__(self, model):
        self.client = OpenAI()
        self.model = model
        self.messages = []
    def forward(self, x):
        self.messages.append({"role": "user", "content": x})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=2000,
            temperature=0.7,
        )
        response = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response
    def get_messages(self):
        return self.messages
    def set_system_prompt(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]
    def clear_messages(self):
        self.messages = []

class Anonymizer:
    def __init__(self, model):
        self.client = OpenAI()
        self.model = model
        self.messages = []
    def forward(self, x):
        self.messages.append({"role": "user", "content": x})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=2000,
            temperature=0.7,
        )
        response = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response
    def get_messages(self):
        return self.messages
    def set_system_prompt(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]
    def clear_messages(self):
        self.messages = []

class Judge:
    def __init__(self, model):
        self.client = OpenAI()
        self.model = model
        self.messages = []
    def forward(self, x):
        self.messages.append({"role": "user", "content": x})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=2000,
            temperature=0.7,
        )
        response = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        return response
    def get_messages(self):
        return self.messages
    def set_system_prompt(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]
    def clear_messages(self):
        self.messages = []
import requests
import json

class LLMOrchestrator:
    def __init__(self, api_key=None):
        self.api_key = api_key 
        self.history = []

    def _call_llm(self, prompt):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": "You are MetaMind, an expert in Computational Intelligence."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            self.history.append({"prompt": prompt, "response": answer})
            return answer
        except requests.exceptions.RequestException as e:
            return f"Network/API error: {e}"
        except (KeyError, IndexError):
            return f"Unexpected API response: {result}"

    def analyze_problem(self, user_input):
        prompt = f"""
        Analyze this problem description: "{user_input}"

        Choose the most appropriate problem type **from these three options only**:
        - "combinatorial_optimization" (e.g., TSP, routing, scheduling)
        - "numerical_optimization" (e.g., function minimization like Rastrigin)
        - "classification" (e.g., predicting categories or classes from data)

        Then, select the best Computational Intelligence method from this list:
        [GA, PSO, ACO, MLP, SOM, Fuzzy, Perceptron, Hopfield, GP]

        IMPORTANT:
        - Output **only** a valid JSON object.
        - Do **not** include any explanations, Markdown formatting, or text outside the JSON.
        - Do **not** wrap your output in ```json or ``` fences.
        - Do **not** include comments inside the JSON.

        Return strictly this JSON structure:
        {{
        "problem_type": "combinatorial_optimization" | "numerical_optimization" | "classification" | "clustering,
        "selected_method": "GA" | "PSO" | "ACO" | "MLP" | "SOM" | "Fuzzy" | "Perceptron" | "Hopfield" | "GP",
        "reasoning": "Explain briefly why this method suits the problem type.",
        "parameters": {{
            // optional hyperparameters like learning_rate, population_size, etc.
        }}
        }}
        """

        text = self._call_llm(prompt)
        print(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print("ERROR")
            return {
                "problem_type": "classification",
                "selected_method": "MLP",
                "reasoning": "Fallback due to parse error",
                "parameters": {}
            }

    def interpret_results(self, results):
        prompt = f"Analyze these results: {results} and give feedback."
        return self._call_llm(prompt)
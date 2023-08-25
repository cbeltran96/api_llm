import flask
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# template = """Question: {question}

# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

template = """
Pregunta: {question}

# Respuesta: Aqui tienes una respuesta corta a tu pregunta en base a mis conocimientos.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Verbose is required to pass to the callback manager

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="ggml-model-q4_0.bin",
    callback_manager=callback_manager,
    verbose=True,
    max_tokens=1000
)

chain = LLMChain(llm=llm,prompt = prompt)


# Crear una aplicación Flask
app = flask.Flask(__name__)

# Crear una ruta para la API
@app.route("/", methods=["POST"])
def predict():
    # Obtener la consulta del usuario
    prompt = flask.request.get_json().get("prompt")

    # Predecir la respuesta
    response = chain.run(prompt)
    print(response)
    # Devolver la respuesta
    return flask.jsonify({"response": response})

# Iniciar la aplicación Flask
app.run()
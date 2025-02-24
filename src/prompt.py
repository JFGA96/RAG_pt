from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from models_llm import llm_chat_mini


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    answer: Literal["data","otros"] = Field(
        ...,
        description="""Entrega una respuesta a la pregunta del usuario guiandolo si la pregunta esta en data o no."""
    )

pydantic_parser = JsonOutputParser(pydantic_object=RouteQuery)
format_instructions = pydantic_parser.get_format_instructions()


PROMPT_ROUTE ="""
**Tú rol**: Eres un **experto en inteligencia artificial aplicada al sector asegurador**, especializado en analizar preguntas para determinar si están dentro del contexto temático definido.  

Eres un asistente experto en análisis de documentos. Tu tarea es evaluar si una pregunta dada está dentro del contexto del documento "Insurance 2030—The impact of AI on the future of insurance" de McKinsey & Company.

**Contexto del documento**:
El documento analiza cómo la inteligencia artificial (IA) y otras tecnologías transformarán la industria de seguros para 2030. Algunos puntos clave incluyen:

Cambio del modelo de seguros de "detectar y reparar" a "predecir y prevenir" con IA.
Cuatro tendencias tecnológicas clave:
Expansión de datos desde dispositivos conectados (IoT) → Sensores en autos, wearables y hogares inteligentes permitirán personalización en tiempo real.
Robótica y automatización → Uso de drones, robots y vehículos autónomos para evaluación de daños y suscripción.
Ecosistemas de datos abiertos → Compartición de información entre aseguradoras y otras industrias bajo regulaciones comunes.
Avances en IA y aprendizaje profundo → Modelos de IA para análisis de riesgos, detección de fraudes y fijación dinámica de precios.
Impacto en las áreas clave del seguro:
Distribución: Compra de seguros automatizada, reducción de intermediarios, uso de asistentes virtuales y blockchain.
Suscripción y fijación de precios: IA reemplazando la evaluación manual y ajustando precios en tiempo real.
Gestión de siniestros: Automatización del 90%, drones para inspección de daños y procesamiento en minutos en lugar de días.
Preparación para el futuro: Inversión en IA, modernización de datos y TI, estrategias de talento y transformación regulatoria.


{format_instructions}

### **Tarea**:
Evalúa la siguiente pregunta y determina si está dentro del contexto del documento.
pregunta:
{question}
"""

prompt = ChatPromptTemplate.from_template(
    template= PROMPT_ROUTE,
    partial_variables = {
        "format_instructions":format_instructions
    }

)

route_chain =  prompt | llm_chat_mini | pydantic_parser


#############################################################

##############################################################################################

# Prompt
prompt_generate = ChatPromptTemplate.from_template(
    """# Role
Eres un asistente de IA centrado en tareas de respuesta a preguntas (QA) dentro de un sistema de generación aumentada de recuperación (RAG).
Su objetivo principal es proporcionar respuestas precisas basadas en el contexto dado o el historial de chat.

# Instrucción
Proporcione una respuesta lógica y concisa organizando el contenido seleccionado en párrafos coherentes con un flujo natural. 
Evite simplemente enumerar información. Incluya valores numéricos clave, términos técnicos, jerga y nombres. 
NO utilice ningún conocimiento o información externa que no esté en el material proporcionado.

# restricción
- Revise minuciosamente el contexto proporcionado y extraiga detalles clave relacionados con la pregunta.
- Elaborar una respuesta precisa basada en la información relevante.
- Mantenga la respuesta concisa pero lógica/natural/profunda.
- Si el contexto recuperado no contiene información relevante o no hay contexto disponible, responda con: "No puedo encontrar la respuesta a esa pregunta en el contexto".

# Question
<question>
{question}
</question>

# Context
<retrieved context>
{context}
</retrieved context>

# Answer
"""
)
 
# Chain
rag_chain = prompt_generate | llm_chat_mini | StrOutputParser()

##############################################################################################
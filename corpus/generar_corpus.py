"""Generador del corpus sintetico multilingue de intenciones de WhatsApp.

Reproduce la metodologia descrita en el primer informe (P1): plantillas por
idioma con *slot-filling* sobre vocabularios especificos y una capa de ruido
superficial controlado (minusculas, eliminacion de tildes, abreviaturas tipo
SMS, insercion de emojis y fragmentos de cortesia).

Salida:  corpus.csv  con columnas  [text, intent, lang]
         12 intenciones x 110 mensajes = 1320 filas
         distribucion  ~70% ES / ~20% EN / ~10% PT

Uso:  python generar_corpus.py
"""
import csv
import os
import random
import unicodedata

RANDOM_STATE = 42
PER_CLASS = 110
# reparto por idioma dentro de cada clase (suma = PER_CLASS)
LANG_SPLIT = {"es": 77, "en": 22, "pt": 11}
# fraccion de cada clase que proviene de plantillas ambiguas (frontera lexica)
AMBIG_RATIO = 0.30

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus.csv")

INTENTS = [
    "constituir_empresa", "comprar_pizza", "reservar_hotel", "comprar_vuelo",
    "consultar_medicina", "agendar_cita_medica", "consulta_banca",
    "soporte_tecnico", "reclamo", "seguimiento_pedido",
    "saludo", "despedida",
]

# --------------------------------------------------------------------------
# Vocabularios de slots por idioma
# --------------------------------------------------------------------------
SLOTS = {
    "es": {
        "ciudad": ["Lima", "Arequipa", "Cusco", "Trujillo", "Piura", "Bogota",
                    "Medellin", "Santiago", "Ciudad de Mexico", "Quito"],
        "pizza": ["hawaiana", "americana", "pepperoni", "cuatro quesos",
                   "vegetariana", "napolitana", "de peperoni", "criolla"],
        "sintoma": ["dolor de cabeza", "fiebre alta", "tos seca", "dolor de estomago",
                     "gripe", "dolor de garganta", "nauseas", "dolor de espalda"],
        "medicina": ["paracetamol", "ibuprofeno", "amoxicilina", "loratadina",
                      "omeprazol", "aspirina", "naproxeno"],
        "especialidad": ["cardiologia", "pediatria", "dermatologia", "traumatologia",
                          "ginecologia", "odontologia", "oftalmologia"],
        "producto": ["la laptop", "el celular", "las zapatillas", "el audifono",
                      "la television", "el teclado", "la impresora"],
        "empresa": ["una SAC", "una SRL", "mi startup", "una EIRL", "mi negocio",
                     "una empresa de servicios"],
        "aerolinea": ["LATAM", "Sky", "Avianca", "Copa", "JetSmart"],
        "hotel": ["un hotel 4 estrellas", "un hostal", "un hotel boutique",
                   "un resort", "un hotel cerca del centro"],
    },
    "en": {
        "ciudad": ["Lima", "Miami", "New York", "Madrid", "Bogota", "Sao Paulo",
                    "Santiago", "Panama City", "Cancun"],
        "pizza": ["hawaiian", "pepperoni", "four cheese", "veggie", "margherita",
                   "bbq chicken"],
        "sintoma": ["a headache", "high fever", "a dry cough", "a stomach ache",
                     "the flu", "a sore throat", "back pain"],
        "medicina": ["paracetamol", "ibuprofen", "amoxicillin", "loratadine",
                      "omeprazole", "aspirin"],
        "especialidad": ["cardiology", "pediatrics", "dermatology", "orthopedics",
                          "gynecology", "dentistry"],
        "producto": ["the laptop", "the phone", "the sneakers", "the headphones",
                      "the tv", "the keyboard"],
        "empresa": ["an LLC", "a startup", "a small business", "a company"],
        "aerolinea": ["LATAM", "Delta", "American", "Copa", "United"],
        "hotel": ["a 4 star hotel", "a hostel", "a boutique hotel", "a resort"],
    },
    "pt": {
        "ciudad": ["Sao Paulo", "Rio de Janeiro", "Lima", "Lisboa", "Bogota",
                    "Santiago", "Brasilia"],
        "pizza": ["havaiana", "calabresa", "quatro queijos", "vegetariana",
                   "portuguesa", "margherita"],
        "sintoma": ["dor de cabeca", "febre alta", "tosse seca", "dor de estomago",
                     "gripe", "dor de garganta"],
        "medicina": ["paracetamol", "ibuprofeno", "amoxicilina", "loratadina",
                      "omeprazol"],
        "especialidad": ["cardiologia", "pediatria", "dermatologia", "ortopedia",
                          "ginecologia"],
        "producto": ["o notebook", "o celular", "os tenis", "o fone", "a tv"],
        "empresa": ["uma empresa", "uma startup", "um pequeno negocio", "uma MEI"],
        "aerolinea": ["LATAM", "Gol", "Azul", "Copa"],
        "hotel": ["um hotel 4 estrelas", "um hostel", "um hotel boutique",
                   "um resort"],
    },
}

# --------------------------------------------------------------------------
# Plantillas por intencion e idioma.  Cada {slot} se rellena desde SLOTS.
# --------------------------------------------------------------------------
TEMPLATES = {
    "constituir_empresa": {
        "es": [
            "Quiero constituir {empresa}, que tramites necesito?",
            "Como registro {empresa} en {ciudad}?",
            "Necesito ayuda para formalizar {empresa}",
            "Cuanto cuesta crear {empresa}?",
            "Quisiera abrir {empresa}, por donde empiezo?",
            "Que documentos piden para constituir {empresa}?",
            "Deseo inscribir {empresa} en registros publicos",
            "Me pueden asesorar para constituir {empresa} en {ciudad}?",
            "Cuanto demora registrar {empresa}?",
            "Quiero formalizar mi negocio y sacar RUC para {empresa}",
        ],
        "en": [
            "I want to set up {empresa}, what do I need?",
            "How do I register {empresa} in {ciudad}?",
            "I need help incorporating {empresa}",
            "How much does it cost to create {empresa}?",
            "Can you help me start {empresa}?",
            "What documents are required to register {empresa}?",
            "I would like to open {empresa}, where do I begin?",
        ],
        "pt": [
            "Quero abrir {empresa}, quais os tramites?",
            "Como registro {empresa} em {ciudad}?",
            "Preciso de ajuda para formalizar {empresa}",
            "Quanto custa criar {empresa}?",
            "Gostaria de abrir {empresa}, por onde comeco?",
        ],
    },
    "comprar_pizza": {
        "es": [
            "Quiero pedir una pizza {pizza}",
            "Me manda una pizza {pizza} a {ciudad}?",
            "Cuanto cuesta la pizza {pizza} familiar?",
            "Quisiera ordenar dos pizzas {pizza}",
            "Tienen pizza {pizza}? quiero pedir una",
            "Haganme delivery de una pizza {pizza} porfa",
            "Deseo una pizza {pizza} grande con extra queso",
            "Pizza {pizza} para llevar, cuanto seria?",
            "Quiero comprar una pizza {pizza} ahora",
            "Me antojo una {pizza}, tienen promo?",
        ],
        "en": [
            "I want to order a {pizza} pizza",
            "Can you deliver a {pizza} pizza to {ciudad}?",
            "How much is a large {pizza} pizza?",
            "I would like two {pizza} pizzas",
            "Do you have {pizza} pizza? I want to order one",
            "Id like a {pizza} pizza for pickup",
            "Can I get a {pizza} pizza with extra cheese?",
        ],
        "pt": [
            "Quero pedir uma pizza {pizza}",
            "Voces entregam pizza {pizza} em {ciudad}?",
            "Quanto custa a pizza {pizza} grande?",
            "Gostaria de uma pizza {pizza}",
            "Tem pizza {pizza}? quero pedir uma",
        ],
    },
    "reservar_hotel": {
        "es": [
            "Quiero reservar {hotel} en {ciudad}",
            "Tienen habitaciones disponibles en {ciudad} para el finde?",
            "Necesito {hotel} para 3 noches en {ciudad}",
            "Cuanto cuesta una habitacion doble en {ciudad}?",
            "Quisiera hacer una reserva en {hotel}",
            "Hay disponibilidad en {hotel} para el viernes?",
            "Deseo reservar {hotel} con desayuno incluido",
            "Me reservan una habitacion en {ciudad} del 10 al 15?",
            "Busco {hotel} economico en {ciudad}",
        ],
        "en": [
            "I want to book {hotel} in {ciudad}",
            "Do you have rooms available in {ciudad} this weekend?",
            "I need {hotel} for 3 nights in {ciudad}",
            "How much is a double room in {ciudad}?",
            "Id like to make a reservation at {hotel}",
            "Is there availability at {hotel} for friday?",
        ],
        "pt": [
            "Quero reservar {hotel} em {ciudad}",
            "Tem quartos disponiveis em {ciudad} no fim de semana?",
            "Preciso de {hotel} por 3 noites em {ciudad}",
            "Quanto custa um quarto duplo em {ciudad}?",
            "Gostaria de fazer uma reserva em {hotel}",
        ],
    },
    "comprar_vuelo": {
        "es": [
            "Quiero comprar un vuelo a {ciudad}",
            "Cuanto cuesta un pasaje a {ciudad} con {aerolinea}?",
            "Busco vuelos baratos a {ciudad}",
            "Necesito un vuelo ida y vuelta a {ciudad}",
            "Quisiera reservar un vuelo con {aerolinea} a {ciudad}",
            "Hay vuelos a {ciudad} para el sabado?",
            "Deseo comprar dos boletos a {ciudad}",
            "Me consigues un pasaje a {ciudad} para manana?",
            "Cotizame un vuelo a {ciudad} en {aerolinea}",
        ],
        "en": [
            "I want to buy a flight to {ciudad}",
            "How much is a ticket to {ciudad} with {aerolinea}?",
            "Im looking for cheap flights to {ciudad}",
            "I need a round trip flight to {ciudad}",
            "Id like to book a flight with {aerolinea} to {ciudad}",
            "Are there flights to {ciudad} on saturday?",
        ],
        "pt": [
            "Quero comprar uma passagem para {ciudad}",
            "Quanto custa um voo para {ciudad} com {aerolinea}?",
            "Procuro voos baratos para {ciudad}",
            "Preciso de um voo ida e volta para {ciudad}",
            "Gostaria de reservar um voo com {aerolinea} para {ciudad}",
        ],
    },
    "consultar_medicina": {
        "es": [
            "Que medicina puedo tomar para {sintoma}?",
            "Tengo {sintoma}, que me recomiendan?",
            "El {medicina} sirve para {sintoma}?",
            "Cuantas veces al dia debo tomar {medicina}?",
            "Que puedo tomar si tengo {sintoma}?",
            "Es seguro tomar {medicina} para {sintoma}?",
            "Que dosis de {medicina} debo tomar?",
            "Tengo {sintoma} desde ayer, que hago?",
            "Me duele mucho, tengo {sintoma}, que medicamento tomo?",
        ],
        "en": [
            "What medicine can I take for {sintoma}?",
            "I have {sintoma}, what do you recommend?",
            "Does {medicina} help with {sintoma}?",
            "How many times a day should I take {medicina}?",
            "What can I take if I have {sintoma}?",
            "Is it safe to take {medicina} for {sintoma}?",
        ],
        "pt": [
            "Que remedio posso tomar para {sintoma}?",
            "Tenho {sintoma}, o que recomendam?",
            "O {medicina} serve para {sintoma}?",
            "Quantas vezes ao dia devo tomar {medicina}?",
            "O que posso tomar se tenho {sintoma}?",
        ],
    },
    "agendar_cita_medica": {
        "es": [
            "Quiero agendar una cita con {especialidad}",
            "Necesito una cita medica de {especialidad} para esta semana",
            "Cuanto cuesta una consulta de {especialidad}?",
            "Me pueden dar hora con el doctor de {especialidad}?",
            "Quisiera reservar una cita de {especialidad} en {ciudad}",
            "Tienen disponibilidad para una consulta de {especialidad}?",
            "Deseo agendar control con {especialidad} para el lunes",
            "A que hora atienden en {especialidad}? quiero sacar cita",
            "Necesito ver a un especialista en {especialidad}, hay citas?",
        ],
        "en": [
            "I want to schedule an appointment with {especialidad}",
            "I need a {especialidad} appointment this week",
            "How much is a {especialidad} consultation?",
            "Can I book a slot with the {especialidad} doctor?",
            "Id like to schedule a {especialidad} visit in {ciudad}",
            "Do you have availability for a {especialidad} appointment?",
        ],
        "pt": [
            "Quero agendar uma consulta de {especialidad}",
            "Preciso de uma consulta de {especialidad} esta semana",
            "Quanto custa uma consulta de {especialidad}?",
            "Posso marcar horario com o medico de {especialidad}?",
            "Gostaria de agendar uma consulta de {especialidad} em {ciudad}",
        ],
    },
    "consulta_banca": {
        "es": [
            "Cual es el saldo de mi cuenta?",
            "Quiero consultar los movimientos de mi cuenta",
            "No puedo transferir dinero desde mi cuenta",
            "Como bloqueo mi tarjeta de credito?",
            "Cuanto tengo disponible en mi cuenta de ahorros?",
            "Quiero pagar mi tarjeta, cuanto debo?",
            "Necesito el estado de cuenta de este mes",
            "Mi cuenta no me deja hacer pagos, que pasa?",
            "Como solicito un aumento de linea de credito?",
            "Quiero saber el limite de mi tarjeta",
        ],
        "en": [
            "What is the balance of my account?",
            "I want to check my account transactions",
            "I cant transfer money from my account",
            "How do I block my credit card?",
            "How much do I have in my savings account?",
            "I want to pay my card, how much do I owe?",
            "I need this months account statement",
        ],
        "pt": [
            "Qual e o saldo da minha conta?",
            "Quero consultar os movimentos da minha conta",
            "Nao consigo transferir dinheiro da minha conta",
            "Como bloqueio meu cartao de credito?",
            "Quanto tenho na minha conta poupanca?",
        ],
    },
    "soporte_tecnico": {
        "es": [
            "Mi cuenta de la app no funciona",
            "No puedo iniciar sesion en la aplicacion",
            "La app se cierra sola cada vez que la abro",
            "El sistema me da error al cargar la pagina",
            "No me llega el codigo de verificacion",
            "La aplicacion esta muy lenta, no carga",
            "No puedo actualizar la app, me sale error",
            "Se me congela la pantalla en la aplicacion",
            "Olvide mi contrasena y no puedo recuperarla",
            "El boton de pago no responde en la web",
        ],
        "en": [
            "My app account is not working",
            "I cant log in to the application",
            "The app keeps crashing when I open it",
            "The system gives me an error loading the page",
            "Im not getting the verification code",
            "The app is very slow and wont load",
            "I forgot my password and cant reset it",
        ],
        "pt": [
            "Minha conta do app nao funciona",
            "Nao consigo fazer login no aplicativo",
            "O app fecha sozinho quando abro",
            "O sistema da erro ao carregar a pagina",
            "Nao recebo o codigo de verificacao",
        ],
    },
    "reclamo": {
        "es": [
            "Quiero poner un reclamo, {producto} llego dañado",
            "Estoy muy molesto, el servicio fue pesimo",
            "Deseo presentar una queja formal por mi pedido",
            "{producto} que compre no funciona, exijo solucion",
            "Me cobraron de mas y quiero un reclamo",
            "El producto llego roto, quiero mi devolucion",
            "Pesimo servicio, quiero hablar con un supervisor",
            "Mi pedido nunca llego y quiero reclamar",
            "Quiero un reembolso, {producto} vino defectuoso",
        ],
        "en": [
            "I want to file a complaint, {producto} arrived damaged",
            "Im very upset, the service was terrible",
            "I want to submit a formal complaint about my order",
            "{producto} I bought doesnt work, I demand a solution",
            "I was overcharged and I want to complain",
            "The product arrived broken, I want a refund",
        ],
        "pt": [
            "Quero fazer uma reclamacao, {producto} chegou danificado",
            "Estou muito irritado, o servico foi pessimo",
            "Desejo apresentar uma queixa formal sobre meu pedido",
            "{producto} que comprei nao funciona, exijo solucao",
            "Meu pedido nunca chegou e quero reclamar",
        ],
    },
    "seguimiento_pedido": {
        "es": [
            "Donde esta mi pedido?",
            "Quiero rastrear mi orden numero 4821",
            "Cuando llega mi pedido de {producto}?",
            "Mi paquete aun no llega, donde esta?",
            "Necesito el estado de mi envio",
            "Ya pague pero no se cuando llega mi pedido",
            "Me pueden dar el tracking de mi orden?",
            "Mi pedido dice en camino hace 3 dias, que paso?",
            "Cuanto falta para que llegue mi compra?",
        ],
        "en": [
            "Where is my order?",
            "I want to track my order number 4821",
            "When does my {producto} order arrive?",
            "My package hasnt arrived yet, where is it?",
            "I need the status of my shipment",
            "Can you give me the tracking for my order?",
        ],
        "pt": [
            "Onde esta o meu pedido?",
            "Quero rastrear meu pedido numero 4821",
            "Quando chega meu pedido de {producto}?",
            "Meu pacote ainda nao chegou, onde esta?",
            "Preciso do status do meu envio",
        ],
    },
    "saludo": {
        "es": [
            "Hola, buenos dias",
            "Buenas tardes, como estan?",
            "Hola que tal",
            "Buenas, hay alguien?",
            "Hola, buenas noches",
            "Que tal, buen dia",
            "Hola, como va todo?",
            "Saludos, buenos dias a todos",
            "Hey hola",
            "Buenas, un gusto saludarlos",
        ],
        "en": [
            "Hi, good morning",
            "Good afternoon, how are you?",
            "Hello there",
            "Hi, is anyone there?",
            "Hey, good evening",
            "Hello, how is it going?",
            "Good morning everyone",
        ],
        "pt": [
            "Ola, bom dia",
            "Boa tarde, como vao?",
            "Ola, tudo bem?",
            "Boa noite",
            "Oi, tudo certo?",
        ],
    },
    "despedida": {
        "es": [
            "Gracias, hasta luego",
            "Ok muchas gracias, chau",
            "Perfecto, nos vemos",
            "Gracias por la ayuda, adios",
            "Listo, hasta pronto",
            "Muy amable, que tenga buen dia",
            "Eso seria todo, gracias y chau",
            "Ok gracias, bye",
            "Nos vemos, cuidate",
            "Gracias, hasta la proxima",
        ],
        "en": [
            "Thanks, see you later",
            "Ok thank you, bye",
            "Perfect, goodbye",
            "Thanks for the help, bye",
            "Alright, see you soon",
            "Thank you, have a nice day",
            "That would be all, thanks and bye",
        ],
        "pt": [
            "Obrigado, ate logo",
            "Ok muito obrigado, tchau",
            "Perfeito, ate mais",
            "Obrigado pela ajuda, adeus",
            "Valeu, ate a proxima",
        ],
    },
}

# --------------------------------------------------------------------------
# Plantillas AMBIGUAS: comparten vocabulario con una clase vecina y crean la
# frontera lexica que hace realista la tarea (reproduce las confusiones
# reportadas en P1: agendar<->consultar_medicina, banca<->soporte_tecnico).
# --------------------------------------------------------------------------
AMBIG = {
    # citas planteadas como preguntas de precio/informacion medica
    "agendar_cita_medica": {
        "es": [
            "Cuanto cuesta una consulta de {especialidad}?",
            "Necesito consulta de psicologia, cuanto sale?",
            "Que precio tiene ver a un {especialidad}?",
            "Atienden {especialidad} manana? cuanto cobran?",
            "Cuanto vale la consulta con el {especialidad}?",
            "Consulta de {especialidad} para mi hijo, cuanto es?",
        ],
        "en": [
            "How much is a {especialidad} consultation?",
            "Do you have availability for {especialidad}? whats the price?",
            "What does it cost to see a {especialidad}?",
        ],
        "pt": [
            "Quanto custa uma consulta de {especialidad}?",
            "Qual o preco da consulta de {especialidad}?",
        ],
    },
    # medicina redactada como si buscara atencion / mencionando ir al doctor
    "consultar_medicina": {
        "es": [
            "Tengo {sintoma}, debo ir al doctor o tomo algo?",
            "Para {sintoma} me tomo {medicina} o mejor consulto?",
            "Que hago con este {sintoma}, necesito medicina?",
            "Mi hijo tiene {sintoma}, que le doy?",
        ],
        "en": [
            "I have {sintoma}, should I see a doctor or take something?",
            "For {sintoma} should I take {medicina} or get checked?",
        ],
        "pt": [
            "Tenho {sintoma}, devo ir ao medico ou tomo algo?",
        ],
    },
    # banca y soporte comparten el mismo pool "mi cuenta ..." SIN pistas que
    # revelen si es cuenta bancaria o cuenta de la aplicacion: frontera real.
    "consulta_banca": {"es": None, "en": None, "pt": None},
    "soporte_tecnico": {"es": None, "en": None, "pt": None},
}

# Pools compartidos por pares confundibles (misma superficie, distinta etiqueta).
CUENTA_SHARED = {
    "es": [
        "Mi cuenta no funciona",
        "No puedo acceder a mi cuenta",
        "No puedo entrar a mi cuenta",
        "Mi cuenta esta bloqueada",
        "Mi cuenta me da error",
        "No me deja iniciar sesion en mi cuenta",
    ],
    "en": [
        "My account is not working",
        "I cant access my account",
        "My account is blocked",
        "I cant log into my account",
    ],
    "pt": [
        "Minha conta nao funciona",
        "Nao consigo acessar minha conta",
        "Minha conta esta bloqueada",
    ],
}
# agendar y consultar_medicina comparten preguntas de precio/consulta medica.
CONSULTA_MED_SHARED = {
    "es": [
        "Cuanto cuesta una consulta de {especialidad}?",
        "Necesito una consulta de {especialidad}",
        "Consulta de {especialidad}, cuanto seria?",
        "Quiero una consulta de {especialidad} para mi hijo",
    ],
    "en": [
        "How much is a {especialidad} consultation?",
        "I need a {especialidad} consultation",
    ],
    "pt": [
        "Quanto custa uma consulta de {especialidad}?",
    ],
}
# asignar los pools compartidos a ambos miembros de cada par
AMBIG["consulta_banca"] = CUENTA_SHARED
AMBIG["soporte_tecnico"] = CUENTA_SHARED
AMBIG["agendar_cita_medica"] = CONSULTA_MED_SHARED
AMBIG["consultar_medicina"] = CONSULTA_MED_SHARED

# --------------------------------------------------------------------------
# Capa de ruido superficial
# --------------------------------------------------------------------------
EMOJIS = ["", "", "", "", "😊", "🙏", "👍", "🍕", "✈️", "🏨", "❤️", "😀", "🙌"]
CORTESIA = {
    "es": ["", "", "porfa", "por favor", "gracias", "urgente", "hola"],
    "en": ["", "", "please", "thanks", "asap", "hi"],
    "pt": ["", "", "por favor", "obrigado", "urgente"],
}
ABBR = {"que": "q", "por": "x", "porque": "xq", "para": "pa", "por favor": "xfa",
        "you": "u", "are": "r", "please": "pls", "for": "4", "to": "2",
        "voce": "vc", "voces": "vcs", "tambem": "tb"}


def quitar_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn")


def meter_typo(t, rng):
    """Introduce un typo simple (swap, drop o duplicado de un caracter)."""
    if len(t) < 6:
        return t
    i = rng.randrange(1, len(t) - 1)
    modo = rng.random()
    if modo < 0.34 and t[i] != " " and t[i + 1] != " ":
        t = t[:i] + t[i + 1] + t[i] + t[i + 2:]        # swap
    elif modo < 0.67 and t[i] != " ":
        t = t[:i] + t[i + 1:]                            # drop
    elif t[i] != " ":
        t = t[:i] + t[i] + t[i:]                         # duplicado
    return t


def aplicar_ruido(texto, lang, rng):
    """Aplica una combinacion aleatoria de transformaciones de superficie."""
    t = texto
    # 1. minusculas (60%)
    if rng.random() < 0.6:
        t = t.lower()
    # 2. quitar tildes (50%)
    if rng.random() < 0.5:
        t = quitar_tildes(t)
    # 3. abreviaturas SMS (35%)
    if rng.random() < 0.35:
        for full, ab in ABBR.items():
            if rng.random() < 0.5:
                t = t.replace(" " + full + " ", " " + ab + " ")
    # 4. quitar signos de apertura (40%)
    if rng.random() < 0.4:
        t = t.replace("¿", "").replace("¡", "")
    # 5. fragmento de cortesia al final (30%)
    if rng.random() < 0.3:
        frag = rng.choice(CORTESIA[lang])
        if frag:
            t = t + " " + frag
    # 6. emoji al final (25%)
    if rng.random() < 0.25:
        em = rng.choice(EMOJIS)
        if em:
            t = t + " " + em
    # 7. typo de caracter (20%)
    if rng.random() < 0.20:
        t = meter_typo(t, rng)
    return t.strip()


def rellenar(template, lang, rng):
    """Rellena los slots {..} de una plantilla."""
    out = template
    for slot, valores in SLOTS[lang].items():
        marca = "{" + slot + "}"
        while marca in out:
            out = out.replace(marca, rng.choice(valores), 1)
    return out


def generar_para(intent, lang, n, rng):
    base_tpl = TEMPLATES[intent][lang]
    amb_tpl = AMBIG.get(intent, {}).get(lang, [])
    n_amb = int(round(n * AMBIG_RATIO)) if amb_tpl else 0
    n_base = n - n_amb

    def muestrear(plantillas, k):
        vistos, filas, intentos = set(), [], 0
        while len(filas) < k and intentos < k * 300:
            intentos += 1
            texto = aplicar_ruido(rellenar(rng.choice(plantillas), lang, rng), lang, rng)
            if texto and texto not in vistos:
                vistos.add(texto)
                filas.append(texto)
        while len(filas) < k:   # respaldo si no se alcanza unicidad
            filas.append(aplicar_ruido(rellenar(rng.choice(plantillas), lang, rng), lang, rng))
        return filas

    filas = muestrear(base_tpl, n_base)
    if n_amb:
        filas += muestrear(amb_tpl, n_amb)
    rng.shuffle(filas)
    return filas


def main():
    rng = random.Random(RANDOM_STATE)
    filas = []
    for intent in INTENTS:
        for lang, n in LANG_SPLIT.items():
            for texto in generar_para(intent, lang, n, rng):
                filas.append((texto, intent, lang))
    rng.shuffle(filas)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "intent", "lang"])
        w.writerows(filas)
    # resumen
    print(f"Corpus generado: {len(filas)} mensajes -> {OUT}")
    from collections import Counter
    ci = Counter(r[1] for r in filas)
    cl = Counter(r[2] for r in filas)
    print("Por intencion:", dict(ci))
    print("Por idioma:", dict(cl))


if __name__ == "__main__":
    main()

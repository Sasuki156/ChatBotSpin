import tensorflow as tf  # Importa la libreria TensorFlow
from tensorflow.keras.preprocessing.text import Tokenizer  # Importa il Tokenizer per la pre-elaborazione del testo
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Importa la funzione per il padding delle sequenze
from tensorflow.keras.models import Sequential  # Importa il modello sequenziale di Keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional  # Importa i layer necessari per il modello
from tensorflow.keras.optimizers import Adam  # Importa l'ottimizzatore Adam
import numpy as np  # Importa la libreria NumPy per il calcolo numerico
import random  # Importa la libreria random per la generazione di numeri casuali
import re  # Importa la libreria per le espressioni regolari
import sqlite3  # Importa la libreria per la gestione dei database SQLite
import matplotlib.pyplot as plt  # Importa Matplotlib per la visualizzazione dei dati
import seaborn as sns  # Importa Seaborn per la visualizzazione esteticamente gradevole
from datetime import datetime  # Importa la classe datetime per la gestione delle date
import nltk  # Importa la libreria NLTK per l'elaborazione del linguaggio naturale
from nltk.tokenize import word_tokenize  # Importa la funzione per il tokenizing delle parole
from textblob import TextBlob  # Importa TextBlob per il trattamento del testo e la correzione ortografica

# Scarica il tokenizer di NLTK se non è già stato fatto
nltk.download('punkt')  # Scarica il pacchetto 'punkt' per il tokenizing

# Connessione al database SQLite
conn = sqlite3.connect('chatbot_interactions.db')  # Crea una connessione al database SQLite
cursor = conn.cursor()  # Crea un cursore per eseguire operazioni sul database

# Creazione della tabella se non esiste già
cursor.execute('''
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY,  # Colonna per l'ID univoco
    user_input TEXT,  # Colonna per l'input dell'utente
    predicted_intent TEXT,  # Colonna per l'intento previsto
    feedback TEXT,  # Colonna per il feedback dell'utente
    timestamp TEXT  # Colonna per il timestamp dell'interazione
)
''')

# Intent e dati di risposta
intents_data = {
    'saluto': [  # Intento per i saluti
        "ciao", "salve", "buongiorno", "buon pomeriggio", "hey", "saluti",
        "come stai?", "buonasera", "buon giorno", "salutami", "salutazioni"
    ],
    'hotel': [  # Intento per le richieste relative agli hotel
        "hotel", "dove posso alloggiare?", "soggiorno", "alloggio", "camere disponibili",
        "hotel a morano calabro", "dove dormire", "suggerisci un hotel", "prenotare una camera",
        "opzioni di alloggio", "dove posso soggiornare?", "raccomandazione hotel"
    ],
    'ristoranti': [  # Intento per le richieste sui ristoranti
        "ristorante", "cibo", "mangiare", "dove posso mangiare", "cenare",
        "consigli per ristoranti", "ristorante buon cibo", "ristoranti tipici",
        "cucina locale", "dove mangiare", "ristoranti a morano calabro",
        "piatti tipici", "miglior ristorante", "gastronomia", "specialità locali"
    ],
    'attrazioni': [  # Intento per le attrazioni turistiche
        "cosa vedere", "attrazioni", "monumenti storici", "cosa visitare a morano calabro",
        "posti interessanti", "musei", "cultura", "storia del paese", "luoghi storici",
        "tour culturali", "punti di interesse", "posti da vedere", "cose da fare"
    ],
    'eventi': [  # Intento per gli eventi
        "eventi locali", "cosa succede", "manifestazioni", "feste a morano calabro",
        "spettacoli", "eventi stagionali", "eventi in corso", "fiere", "festival",
        "concerti", "attività speciali", "celebrazioni", "feste tradizionali"
    ],
    'attivita': [  # Intento per le attività
        "attività all'aperto", "sport", "escursioni", "passeggiate",
        "cose da fare all'aperto", "attività per famiglie", "avventure",
        "escursioni naturali", "cicloturismo", "camminate", "sport acquatici"
    ],
    'informazioni_pratiche': [  # Intento per le informazioni pratiche
        "informazioni di viaggio", "trasporti", "orari", "ufficio turistico",
        "informazioni utili", "mappa della città", "trasporti locali", "stazioni",
        "bancomat", "farmacie", "pronto soccorso", "numeri utili"
    ],
    'prenotazione': [  # Intento per le prenotazioni
        "prenotare", "come posso prenotare", "effettuare una prenotazione",
        "prenotazione hotel", "prenotazione ristorante", "prenota un'attività",
        "prenotare online", "disponibilità", "prenotazione per tour",
        "costi di prenotazione", "annulla prenotazione"
    ],
    'grazie': [  # Intento per i ringraziamenti
        "grazie", "ti ringrazio", "molte grazie", "grazie mille", "apprezzo il tuo aiuto",
        "sei molto gentile", "grazie infinite", "ti sono grato"
    ],
    'addio': [  # Intento per gli addii
        "arrivederci", "addio", "a presto", "ci vediamo", "fino alla prossima",
        "ciao, grazie", "alla prossima", "ci sentiamo", "buona giornata"
    ]
}

# Risposte migliorate con variazioni
responses = {
    'saluto': [  # Risposte per i saluti
        "Ciao! Benvenuto a Morano Calabro Travel. Come posso aiutarti oggi?",
        "Salve! Siamo qui per aiutarti a scoprire Morano Calabro. Cosa vuoi sapere?"
    ],
    'hotel': [  # Risposte per le richieste sugli hotel
        "A Morano Calabro puoi trovare alloggi come la Locanda del Parco o l'hotel Meruo, con opzioni che variano tra camere economiche e suite di lusso.",
        "Ti consiglio di controllare l'hotel La Fenice, che offre un soggiorno confortevole con una vista spettacolare."
    ],
    'ristoranti': [  # Risposte per le richieste sui ristoranti
        "Per la cucina tipica, prova il ristorante 'Antica Masseria Salmena' o altri locali che offrono piatti tradizionali calabresi.",
        "Puoi anche visitare il ristorante 'Il Girasole', famoso per le sue pizze e piatti a base di pesce."
    ],
    'attrazioni': [  # Risposte per le attrazioni turistiche
        "Morano Calabro offre luoghi storici come il Castello Normanno-Svevo, la Chiesa di San Bernardino e il Museo del Territorio.",
        "Non perdere una passeggiata per le stradine medievali del centro; sono piene di storia e fascino."
    ],
    'eventi': [  # Risposte per gli eventi
        "Gli eventi più conosciuti includono la Festa della Madonna di Costantinopoli a maggio e la Sagra del Peperoncino in estate.",
        "Puoi anche partecipare a concerti e festival locali che si tengono durante tutto l'anno."
    ],
    'attivita': [  # Risposte per le attività
        "Attività popolari includono escursioni al Parco Nazionale del Pollino, cicloturismo e visite guidate.",
        "Per famiglie, sono disponibili anche percorsi naturalistici e attività all'aria aperta."
    ],
    'informazioni_pratiche': [  # Risposte per le informazioni pratiche
        "L'ufficio turistico locale si trova nel centro storico, e puoi ottenere una mappa della città e informazioni su trasporti e orari.",
        "Ci sono anche farmacie, bancomat e altri servizi facilmente accessibili in tutta la città."
    ],
    'prenotazione': [  # Risposte per le prenotazioni
        "Puoi prenotare hotel, ristoranti e attività tramite il nostro sito o presso l'ufficio turistico locale.",
        "È

 possibile prenotare anche telefonicamente, assicurati di verificare la disponibilità."
    ],
    'grazie': [  # Risposte per i ringraziamenti
        "Prego! Sono qui per aiutarti. Fammi sapere se hai altre domande.",
        "Non c'è di che! Se hai bisogno di ulteriore assistenza, non esitare a chiedere."
    ],
    'addio': [  # Risposte per gli addii
        "Arrivederci! Spero di esserti stato utile. A presto!",
        "Ci vediamo! Grazie per averci contattato e spero di rivederti presto."
    ]
}

# Prepara i dati
all_sentences, all_labels = [], []  # Liste per tutte le frasi e le etichette
labels = list(intents_data.keys())  # Ottiene le etichette degli intenti
label_index = {label: idx for idx, label in enumerate(labels)}  # Crea un dizionario che mappa le etichette agli indici

# Estrae le frasi e le etichette
for intent, sentences in intents_data.items():  # Per ogni intento e le sue frasi
    all_sentences.extend(sentences)  # Aggiunge le frasi alla lista di frasi
    all_labels.extend([label_index[intent]] * len(sentences))  # Aggiunge l'etichetta corrispondente

# Tokenizzazione e padding
tokenizer = Tokenizer(oov_token="<OOV>")  # Crea un tokenizer con un token per le parole sconosciute
tokenizer.fit_on_texts(all_sentences)  # Adatta il tokenizer ai dati
total_words = len(tokenizer.word_index) + 1  # Calcola il numero totale di parole uniche

# Converte le frasi in sequenze di numeri
sequences = tokenizer.texts_to_sequences(all_sentences)  # Converte le frasi in sequenze di indici
padded_sequences = pad_sequences(sequences, padding='post')  # Applica il padding alle sequenze

# Creazione del modello
model = Sequential([  # Inizializza un modello sequenziale
    Embedding(input_dim=total_words, output_dim=64, input_length=padded_sequences.shape[1]),  # Layer di embedding
    Bidirectional(LSTM(128, return_sequences=True)),  # Layer LSTM bidirezionale
    Dropout(0.3),  # Layer di dropout per prevenire l'overfitting
    LSTM(64),  # Secondo layer LSTM
    Dense(64, activation='relu'),  # Layer denso con attivazione ReLU
    Dropout(0.3),  # Ulteriore layer di dropout
    Dense(len(labels), activation='softmax')  # Layer finale per la classificazione degli intenti
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compila il modello con l'ottimizzatore e la funzione di perdita

# Addestramento del modello
history = model.fit(padded_sequences, np.array(all_labels), epochs=300, batch_size=16)  # Addestra il modello

# Funzione per tracciare e visualizzare le metriche in modo esteticamente gradevole
def plot_metrics(history):
    sns.set(style='whitegrid')  # Imposta lo stile di Seaborn
    plt.figure(figsize=(14, 7))  # Imposta la dimensione della figura

    # Plot per la Loss
    plt.subplot(1, 2, 1)  # Crea un subplot
    plt.plot(history.history['loss'], label='Loss', color='red', linewidth=2)  # Plotta la perdita
    plt.title('Model Loss', fontsize=16)  # Titolo del grafico
    plt.xlabel('Epochs', fontsize=14)  # Etichetta dell'asse x
    plt.ylabel('Loss', fontsize=14)  # Etichetta dell'asse y
    plt.xticks(fontsize=12)  # Imposta la dimensione del font per le etichette dell'asse x
    plt.yticks(fontsize=12)  # Imposta la dimensione del font per le etichette dell'asse y
    plt.legend(fontsize=12)  # Mostra la leggenda
    plt.grid(True)  # Mostra la griglia

    # Plot per l'Accuracy
    plt.subplot(1, 2, 2)  # Crea un secondo subplot
    plt.plot(history.history['accuracy'], label='Accuracy', color='green', linewidth=2)  # Plotta l'accuratezza
    plt.title('Model Accuracy', fontsize=16)  # Titolo del grafico
    plt.xlabel('Epochs', fontsize=14)  # Etichetta dell'asse x
    plt.ylabel('Accuracy', fontsize=14)  # Etichetta dell'asse y
    plt.xticks(fontsize=12)  # Imposta la dimensione del font per le etichette dell'asse x
    plt.yticks(fontsize=12)  # Imposta la dimensione del font per le etichette dell'asse y
    plt.legend(fontsize=12)  # Mostra la leggenda
    plt.grid(True)  # Mostra la griglia

    plt.tight_layout()  # Ottimizza il layout
    plt.savefig('model_metrics.png', dpi=300)  # Salva il grafico delle metriche
    plt.show()  # Mostra il grafico

plot_metrics(history)  # Chiama la funzione per visualizzare le metriche

# Funzioni di pre-processing e tokenizzazione
def preprocess_input(user_input):  # Funzione per pre-elaborare l'input dell'utente
    user_input = user_input.lower()  # Converte l'input in minuscolo
    user_input = re.sub(r'[^\w\s]', '', user_input)  # Rimuove la punteggiatura
    return user_input  # Restituisce l'input pre-elaborato

def correct_spelling(user_input):  # Funzione per correggere l'ortografia
    corrected_input = str(TextBlob(user_input).correct())  # Corregge l'input usando TextBlob
    return corrected_input  # Restituisce l'input corretto

def extract_keywords(user_input):  # Funzione per estrarre le parole chiave
    tokens = word_tokenize(user_input)  # Tokenizza l'input dell'utente
    return tokens  # Restituisce i token

def predict_intent(user_input):  # Funzione per prevedere l'intento dell'input dell'utente
    user_input = preprocess_input(user_input)  # Pre-elabora l'input
    corrected_input = correct_spelling(user_input)  # Corregge l'ortografia dell'input
    tokenized_input = tokenizer.texts_to_sequences([corrected_input])  # Converte l'input in sequenze
    padded_input = pad_sequences(tokenized_input, maxlen=padded_sequences.shape[1], padding='post')  # Applica il padding

    predicted_class = model.predict(padded_input)  # Prevede la classe dell'input
    predicted_intent = labels[np.argmax(predicted_class)]  # Ottiene l'intento previsto dall'output del modello
    return predicted_intent  # Restituisce l'intento previsto

print("Benvenuto nel chatbot di Morano Calabro! (Digita 'exit' per uscire)")  # Messaggio di benvenuto

# Ciclo principale del chatbot
while True:  # Inizia un ciclo infinito
    user_input = input("Tu: ")  # Ottiene l'input dell'utente
    if user_input.lower() == 'exit':  # Controlla se l'utente vuole uscire
        break  # Esce dal ciclo

    predicted_intent = predict_intent(user_input)  # Prevede l'intento dell'input dell'utente
    response = random.choice(responses[predicted_intent])  # Seleziona una risposta casuale per l'intento previsto
    
    # Registra l'interazione nel database
    cursor.execute('''  # Esegue un'istruzione SQL per inserire l'interazione nel database
        INSERT INTO interactions (user_input, predicted_intent, feedback, timestamp)
        VALUES (?, ?, ?, ?)  # Inserisce i valori nei campi della tabella
    ''', (user_input, predicted_intent, None, datetime.now().isoformat()))  # Inserisce i valori dell'input, intento previsto, feedback e timestamp
    conn.commit()  # Conferma la transazione nel database

    print(f"Bot: {response}")  # Stampa la risposta del chatbot

# Richiesta di feedback finale dopo che l'utente ha terminato le domande
feedback = input("Hai finito di fare domande. Sei soddisfatto delle risposte fornite? (sì/no): ")  # Chiede feedback all'utente
cursor.execute('''  # Esegue un'istruzione SQL per aggiornare il feedback nel database
    UPDATE interactions SET feedback = ? WHERE id = (SELECT MAX(id) FROM interactions)  # Aggiorna il feedback per l'ultima interazione
''', (feedback,))  # Passa il feedback come parametro
conn.commit()  # Conferma la transazione nel database

# Funzione per mostrare i dati delle interazioni
def show_interaction_data():  # Funzione per mostrare i dati delle interazioni
    cursor.execute('SELECT * FROM interactions')  # Esegue una query per selezionare tutte le interazioni
    rows = cursor.fetchall()  # Recupera tutte le righe del risultato
    for row in rows:  # Itera su ogni riga
        print(row)  # Stampa la riga

show_interaction_data()  # Chiama la funzione per mostrare i dati delle interazioni

conn.close()  # Chiude la connessione al database

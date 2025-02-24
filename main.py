# Import der benötigten Bibliotheken und Module
import logging                      # Für Logging-Zwecke
import torch                        # PyTorch-Bibliothek für Deep Learning
import torch.nn as nn               # Enthält Module und Klassen für neuronale Netze
import torch.nn.functional as F     # Funktionale Schnittstelle für Aktivierungen und mehr
import torch.optim as optim         # Optimierungsalgorithmen wie Adam
from torch.utils.data import DataLoader, Dataset, random_split  # Datenhandling in PyTorch
import re                           # Reguläre Ausdrücke für die Tokenisierung
from dataclasses import dataclass, field, asdict  # Moderne Python Data Classes (ab Python 3.7)
import random                       # Für zufallsbasierte Prozesse
import numpy as np                  # Numerische Berechnungen mit Arrays
import yaml                         # YAML-Format (z.B. für Konfigurationen)
from sklearn.metrics import accuracy_score, f1_score  # Metriken zur Evaluierung
import matplotlib.pyplot as plt     # Plotten von Trainingskurven
import optuna                       # Für Hyperparameter-Optimierung (optional)
from tqdm import tqdm               # Fortschrittsanzeige für Schleifen
import os                           # Betriebssystemfunktionen, z.B. zum Erstellen von Verzeichnissen

# Debug-Modus aktivieren – nützlich während der Entwicklung, um zusätzliche Informationen zu erhalten
DEBUG_MODE = True

# Definition von speziellen Token-Indexen für Padding, unbekannte Tokens, Start- und End-Symbol
PAD_INDEX = 0
UNK_INDEX = 1
SOS_INDEX = 2
EOS_INDEX = 3


# -------------------------------------------------------------------------------------------------------------------
# Konfigurationsklasse: Hier werden alle wichtigen Hyperparameter und Pfade zentral definiert.
# Die Verwendung einer Data Class (seit Python 3.7) sorgt für sauberen und übersichtlichen Code.
@dataclass
class Config:
    data_path: str = "dataset.txt"         # Pfad zur Datendatei
    log_file: str = "training.log"           # Protokolldatei für Trainingsausgaben
    embedding_dim: int = 256                 # Dimension der Einbettungsvektoren
    memory_size: int = 512                   # Größe des differentiable memory
    learning_rate: float = 0.001             # Lernrate für den Optimierer
    batch_size: int = 32                     # Anzahl der Samples pro Batch
    max_seq_length: int = 50                 # Maximale Länge der Sequenzen
    train_size_ratio: float = 0.8            # Anteil der Trainingsdaten
    val_size_ratio: float = 0.1              # Anteil der Validierungsdaten
    epochs: int = 10                         # Anzahl der Trainingsepochen
    accumulation_steps: int = 1              # Schritte für Gradientenakkumulation (bei Bedarf)
    write_strength: float = 0.1              # Stärke des Updates beim Schreiben in den Speicher
    patience: int = 5                        # Geduld für frühes Stoppen (falls implementiert)
    save_path: str = "models/"               # Pfad zum Speichern von Modellen
    load_model: bool = False                 # Flag zum Laden eines bereits gespeicherten Modells
    plot_path: str = "training_plots/"       # Pfad zum Speichern der Trainingsplots
    device: torch.device = field(init=False) # Gerät (CPU oder GPU), wird im __post_init__ festgelegt

    # Nachinitialisierung: Hier wird das passende Gerät ausgewählt und notwendige Verzeichnisse werden erstellt.
    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)


# -------------------------------------------------------------------------------------------------------------------
# DifferentiableMemory: Implementiert einen Speicher, der als Parameter im Modell geführt wird.
# Dies ermöglicht es, während des Trainings differentiable (ableitbare) Lese- und Schreiboperationen durchzuführen.
class DifferentiableMemory(nn.Module):
    def __init__(self, memory_size: int, embedding_size: int, device):
        super().__init__()
        # Initialisiert zufällige Schlüssel und Werte als Parameter (trainierbar)
        self.keys = nn.Parameter(torch.randn(memory_size, embedding_size, device=device))
        self.values = nn.Parameter(torch.randn(memory_size, embedding_size, device=device))
        self.device = device
        self.memory_size = memory_size

    # Leseoperation: Berechnet Ähnlichkeitswerte zwischen Abfrage (query) und den Schlüsseln, 
    # wendet Softmax an und gibt eine gewichtete Summe der Werte zurück.
    def read(self, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, self.keys.T)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, self.values)

    # Schreiboperation: Aktualisiert die Schlüssel und Werte basierend auf neuen Werten.
    # Die Aktualisierungen werden normalisiert und über einen Write-Strength-Faktor gemischt.
    def write(self, updated_keys: torch.Tensor, updated_values: torch.Tensor, write_strength: float):
        updated_keys = F.normalize(updated_keys, dim=-1)
        updated_values = F.normalize(updated_values, dim=-1)
        with torch.no_grad():
            self.keys.copy_((1 - write_strength) * self.keys + write_strength * updated_keys)
            self.values.copy_((1 - write_strength) * self.values + write_strength * updated_values)


# -------------------------------------------------------------------------------------------------------------------
# Controller: Das Hauptmodell, das den differentiable memory, ein Embedding-Layer, ein RNN (GRU) und einen fully-connected
# Layer kombiniert, um Texteingaben in Vorhersagen (Wortwahrscheinlichkeiten) umzuwandeln.
class Controller(nn.Module):
    def __init__(self, embedding_size: int, memory_size: int, vocab_size: int, device):
        super().__init__()
        # Initialisierung des differentiable memory
        self.memory = DifferentiableMemory(memory_size, embedding_size, device)
        # Embedding-Layer wandelt Token-IDs in dichte Vektorrepräsentationen um
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # GRU: Rekurrentes neuronales Netz, hier mit 2 Schichten, um Sequenzinformationen zu verarbeiten
        self.rnn = nn.GRU(embedding_size, 256, num_layers=2, batch_first=True)
        # Fully-connected Layer, der den RNN-Ausgang in Vokabular-Größen (Wortwahrscheinlichkeiten) überführt
        self.fc_output = nn.Linear(256, vocab_size)
        self.device = device

    # Vorwärtsdurchlauf: In diesem Schritt werden die Eingaben zuerst eingebettet, 
    # anschließend wird der differentiable memory gelesen und das RNN verarbeitet die Speicherrepräsentation.
    def forward(self, inputs: torch.Tensor):
        embedded = self.embedding(inputs)
        memory_output = self.memory.read(embedded)
        rnn_output, _ = self.rnn(memory_output)
        return self.fc_output(rnn_output), memory_output


# -------------------------------------------------------------------------------------------------------------------
# Vocabulary: Eine einfache Klasse zur Verwaltung von Token-IDs.
# Neben dem Hinzufügen von Tokens ermöglicht sie auch den Rückgriff auf die Originaltokens anhand des Index.
class Vocabulary:
    def __init__(self, special_tokens=None):
        self.token_to_index = {}
        self.index_to_token = []
        # Falls spezielle Tokens (wie Padding, SOS, EOS) angegeben werden, werden diese zuerst hinzugefügt.
        for token in special_tokens or []:
            self.add_token(token)

    # Fügt ein neues Token hinzu, sofern es noch nicht vorhanden ist.
    def add_token(self, token):
        if token not in self.token_to_index:
            self.token_to_index[token] = len(self.index_to_token)
            self.index_to_token.append(token)

    def __len__(self):
        return len(self.token_to_index)

    # Gibt die ID für ein Token zurück oder den UNK_INDEX, falls das Token nicht im Vokabular ist.
    def __getitem__(self, token):
        return self.token_to_index.get(token, UNK_INDEX)

    # Methode, um anhand des Index das originale Token zurückzugeben.
    def index_to_token_method(self, index): 
        return self.index_to_token[index]


# -------------------------------------------------------------------------------------------------------------------
# Funktion zum Tokenisieren eines Textes.
# Sie wandelt den Text in Kleinbuchstaben um, fügt Leerzeichen um Satzzeichen hinzu und zerlegt den Text in einzelne Tokens.
def create_tokenizer(text):
    text = text.lower()
    text = re.sub(r"([.,?!])", r" \\1 ", text)
    return re.findall(r'\b\w+|\S\b', text)


# -------------------------------------------------------------------------------------------------------------------
# Erzeugt das Vokabular anhand der Datendatei.
# Dabei werden die Zeilen der Datei tokenisiert und alle Tokens dem Vokabular hinzugefügt.
def create_vocabulary(config):
    tokenizer = create_tokenizer
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    vocab = Vocabulary(special_tokens)

    with open(config.data_path, "r", encoding="utf-8") as f:
        for line in f:
            for token in tokenizer(line.strip()):
                vocab.add_token(token)

    return tokenizer, vocab


# -------------------------------------------------------------------------------------------------------------------
# TextDataset: Ein Dataset, das aus einer Textdatei Zeile für Zeile liest.
# Jede Zeile wird tokenisiert und in eine Sequenz von Token-IDs umgewandelt.
class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, vocab, max_seq_length):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        # Lade alle Zeilen der Datendatei in eine Liste
        self.data = [line.strip() for line in open(data_path, encoding="utf-8")]

    def __getitem__(self, idx):
        # Tokenisiere die Zeile und wandle jedes Token in seinen Index um
        tokens = [self.vocab[token] for token in self.tokenizer(self.data[idx])]
        # Füge den SOS-Token am Anfang und den EOS-Token am Ende hinzu
        input_ids = [SOS_INDEX] + tokens[:self.max_seq_length] + [EOS_INDEX]

        # Berechne, wie viel Padding benötigt wird, um die maximale Sequenzlänge zu erreichen
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [PAD_INDEX] * padding_length
        else:
            input_ids = input_ids[:self.max_seq_length]

        # Rückgabe von Eingabe- und Zielsequenz (hier identisch)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)


# -------------------------------------------------------------------------------------------------------------------
# Evaluierungsfunktion für das Modell: Sie berechnet den Verlust, die Genauigkeit und den F1-Score
# über einen übergebenen DataLoader.
def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Schaltet den Evaluierungsmodus ein (z.B. deaktiviert Dropout)
    total_loss, total_accuracy, total_f1 = 0, 0, 0
    with torch.no_grad():  # Keine Gradientenberechnung, um Speicher zu sparen
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            # Berechne den Verlust über die gesamte Batch
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
            # Berechne die Vorhersagen durch Auswahl des Tokens mit der höchsten Wahrscheinlichkeit
            predictions = torch.argmax(outputs, dim=-1)
            total_accuracy += accuracy_score(targets.cpu().flatten(), predictions.cpu().flatten())
            total_f1 += f1_score(targets.cpu().flatten(), predictions.cpu().flatten(), average='weighted', zero_division=1)

    # Durchschnittswerte über alle Batches
    return total_loss / len(dataloader), total_accuracy / len(dataloader), total_f1 / len(dataloader)


# -------------------------------------------------------------------------------------------------------------------
# Funktion zum Plotten und Speichern der Trainings- und Validierungsmetriken.
def plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores, plot_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot für Verlustwerte
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Trainingsverlust')
    plt.plot(epochs, val_losses, 'r-', label='Validierungsverlust')
    plt.title('Trainings- und Validierungsverlust')
    plt.xlabel('Epochen')
    plt.ylabel('Verlust')
    plt.legend()

    # Plot für Genauigkeit und F1-Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'b-', label='Validierungsgenauigkeit')
    plt.plot(epochs, val_f1_scores, 'r-', label='Validierungs-F1-Score')
    plt.title('Validierungsgenauigkeit und F1-Score')
    plt.xlabel('Epochen')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plot_filename = os.path.join(plot_path, "training_metrics.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Trainingsplots gespeichert in: {plot_filename}")


# -------------------------------------------------------------------------------------------------------------------
# Trainingsfunktion: Hier werden Datensatz, Model, Optimierer und Verlustfunktion erstellt.
# Anschließend wird das Modell trainiert, validiert und der Modellzustand nach jeder Epoche gespeichert.
def train_model(config):
    # Erzeuge Tokenizer und Vokabular basierend auf der Datendatei
    tokenizer, vocab = create_vocabulary(config)
    dataset = TextDataset(config.data_path, tokenizer, vocab, config.max_seq_length)
    # Aufteilen des Datensatzes in Training, Validierung und Test
    train_size = int(config.train_size_ratio * len(dataset))
    val_size = int(config.val_size_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Modellinitialisierung
    model = Controller(config.embedding_dim, config.memory_size, len(vocab), config.device).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    model_path = os.path.join(config.save_path, "model.pth")

    # Optional: Laden eines bereits gespeicherten Modells, falls load_model aktiviert ist.
    if config.load_model and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Modell von Epoche {start_epoch} geladen. Training wird fortgesetzt.")
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}. Training startet von neuem.")
            start_epoch = 0

    # Listen zur Speicherung der Trainings- und Validierungsmesswerte
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []

    # Training über die definierte Anzahl von Epochen
    for epoch in range(start_epoch, config.epochs):
        model.train()  # Schalte den Trainingsmodus ein
        epoch_loss = 0.0
        # Verwende tqdm für eine übersichtliche Fortschrittsanzeige pro Epoche
        with tqdm(train_loader, desc=f"Epoche {epoch + 1}/{config.epochs} (Training)") as t:
            for inputs, targets in t:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                optimizer.zero_grad()  # Gradienten zurücksetzen
                outputs, _ = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()  # Backpropagation
                optimizer.step()  # Aktualisiere die Modellparameter
                epoch_loss += loss.item()
                t.set_postfix(loss=epoch_loss / len(train_loader))

        train_losses.append(epoch_loss / len(train_loader))

        # Evaluierung auf dem Validierungsdatensatz
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion, config.device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"Epoche {epoch + 1}/{config.epochs} (Validierung) - Verlust: {val_loss:.4f}, Genauigkeit: {val_accuracy:.4f}, F1-Score: {val_f1:.4f}")

        # Speichern des Modells nach jeder Epoche
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / len(train_loader),
        }, model_path)
        print(f"Modell gespeichert in: {model_path}")

    # Speichern und Plotten der Trainingsverläufe
    plot_training_history(train_losses, val_losses, val_accuracies, val_f1_scores, config.plot_path)

    # Abschließende Evaluierung auf dem Testdatensatz
    test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader, criterion, config.device)
    print("\n--- Testergebnisse ---")
    print(f"Verlust auf Test-Set: {test_loss:.4f}, Genauigkeit: {test_accuracy:.4f}, F1-Score: {test_f1:.4f}")


# -------------------------------------------------------------------------------------------------------------------
# Funktion zur Textgenerierung:
# Basierend auf einem gegebenen Prompt wird mit dem trainierten Modell eine Textsequenz generiert.
def generate_text(model, vocab, tokenizer, device, prompt_text, max_tokens=30):
    """Generiert Text basierend auf einem Prompt mit dem trainierten Modell."""
    model.eval()  # Schalte in den Evaluationsmodus (z.B. deaktiviert Dropout)
    # Tokenisierung des Prompts und Umwandlung in Token-IDs
    tokens = [vocab[token] for token in tokenizer(prompt_text)]
    # Erstellen des Input-Tensors; füge den SOS-Token hinzu und kürze auf die maximale Länge
    input_tensor = torch.tensor([([SOS_INDEX] + tokens)[:CONFIG.max_seq_length]], dtype=torch.long).to(device)

    generated_tokens = []

    with torch.no_grad():  # Keine Gradientenberechnung während der Generierung
        for _ in range(max_tokens):
            outputs, _ = model(input_tensor)
            # Betrachte nur das letzte Token des Outputs, um den nächsten Token vorherzusagen
            next_token_probs = F.softmax(outputs[:, -1, :], dim=-1)
            # Sampling anstatt argmax: Dies erlaubt eine variablere und kreativere Textgenerierung
            next_token_index = torch.multinomial(next_token_probs, num_samples=1).item()

            # Stoppe die Generierung, wenn das EOS-Token erreicht wird
            if next_token_index == EOS_INDEX:
                break

            generated_tokens.append(next_token_index)
            # Hänge das vorhergesagte Token an den Input an, um Kontext für die nächste Vorhersage zu liefern
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_token_index]], dtype=torch.long).to(device)), dim=1)
            # Beende die Schleife, falls die maximale Sequenzlänge erreicht ist
            if input_tensor.size(1) >= CONFIG.max_seq_length:
                break

    # Dekodiere die generierten Token zurück in lesbaren Text mithilfe der Methode im Vokabular
    generated_text_tokens_decoded = [vocab.index_to_token_method(token) for token in generated_tokens]
    generated_text = " ".join(generated_text_tokens_decoded)
    return generated_text


# -------------------------------------------------------------------------------------------------------------------
# Hauptprogramm: Der Einstiegspunkt des Skripts.
if __name__ == "__main__":
    # Konfiguration initialisieren: Hier wird z.B. die Anzahl der Epochen festgelegt und ob ein Modell geladen werden soll.
    CONFIG = Config(epochs=15, load_model=True)
    train_model(CONFIG)

    # Nach dem Training wird das Modell getestet und einige Beispiel-Queries zur Textgenerierung durchgeführt.
    tokenizer, vocab = create_vocabulary(CONFIG)  # Erstelle Vokabular und Tokenizer neu, um auf dem aktuellen Stand zu sein
    model = Controller(CONFIG.embedding_dim, CONFIG.memory_size, len(vocab), CONFIG.device).to(CONFIG.device)  # Initialisiere das Modell
    model_path = os.path.join(CONFIG.save_path, "model.pth")
    checkpoint = torch.load(model_path)  # Lade den gespeicherten Modellzustand
    model.load_state_dict(checkpoint['model_state_dict'])  # Setze die Modellparameter

    # Definierte Beispiel-Queries, die vom Modell bearbeitet werden sollen
    queries = [
        "Verordnung (EU) 2024/1689 DES EUROPÄISCHEN PARLAMENTS UND DES RATES",
        "Zweck dieser Verordnung ist es",
        "KI-Systeme können problemlos in verschiedenen Bereichen der Wirtschaft und Gesellschaft,",
        "Diese Verordnung sollte im Einklang mit den in der Charta verankerten Werten der Union angewandt werden,"
    ]

    print("\n--- Modell-Queries und Generierungen ---")
    for query in queries:
        generated_response = generate_text(model, vocab, tokenizer, CONFIG.device, query)
        print(f"Query: '{query}'")
        print(f"Generierte Antwort: '{generated_response}'\n")

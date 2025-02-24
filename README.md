# DRLMemNet: Ein Deep Learning Modell mit differenzierbarem Speicher

## Beschreibung

DRLMemNet ist ein Deep-Learning-Modell mit einem integrierten differenzierbaren Speicher, das für Aufgaben der sequenziellen Datenverarbeitung entwickelt wurde, wie z.B. Textgenerierung oder -analyse. Es kombiniert die Fähigkeiten eines sequenziellen Modells (GRU) mit der Fähigkeit, Informationen in einem externen Speicher zu speichern und abzurufen.

### Kernkomponenten

1.  **Differentiable Memory:**
    *   Ein Speicher, der durch **Keys** und **Values** repräsentiert wird. Diese sind als trainierbare Parameter (Tensoren) implementiert.
    *   **Lesen:** Eine Query (Abfrage) wird verwendet, um eine gewichtete Summe von Werten aus dem Speicher zu lesen. Die Gewichte basieren auf der Ähnlichkeit der Query mit den Keys im Speicher (Softmax-Attention).
    *   **Schreiben:** Werte werden in den Speicher mit einem bestimmten Schreib-Stärke-Parameter (`write_strength`) geschrieben, und zwar basierend auf dem Durchschnitt der aktualisierten Keys und Values.
    *   Der Speicher verfolgt die Nutzungshäufigkeit der Speicherplätze.
    *   Die Speicherwerte werden bei jeder Schreiboperation normalisiert.

2.  **Controller:**
    *   Enthält ein `Embedding`-Layer für die Eingabetoken.
    *   Eine `Linear`-Schicht zur Transformation des Embeddings in eine **Query** für den Speicher.
    *   Eine `GRU`-Schicht (Gated Recurrent Unit) zur sequenziellen Verarbeitung.
    *   Mehrere `Linear`- und `Dropout`-Schichten für die Merkmalsanpassung und Regularisierung.
    *   Eine weitere lineare Schicht für die Ausgabe (Vokabulargröße).
    *   Zusätzliche `Linear`-Schichten zur Vorbereitung der Memory-Updates.

3.  **Vocabulary:**
    *   Ein Vokabular, das die Zuordnung zwischen Token und Indizes und umgekehrt verwaltet.
    *   Enthält spezielle Tokens wie `<pad>`, `<unk>`, `<sos>` und `<eos>`.

4.  **TextDataset:**
    *   Ein benutzerdefinierter Dataset-Typ, der Text aus einer Datei lädt, ihn tokenisiert und in numerische Indizes umwandelt.
    *   Führt Padding auf die maximale Sequenzlänge aus.

### Architektur

Die Kernarchitektur ist wie folgt:

1.  Die Eingabetoken werden eingebettet.
2.  Die eingebetteten Eingaben werden in eine Query für den Speicher transformiert.
3.  Die Query wird verwendet, um einen gewichteten Value aus dem Speicher zu lesen.
4.  Der Speicheroutput wird an eine GRU-Schicht weitergeleitet, um die Sequenz zu verarbeiten.
5.  Der RNN-Output wird durch weitere lineare Schichten transformiert.
6.  Eine Ausgabeschicht mit Softmax-Aktivierung liefert die Tokenvorhersagen.
7.  Die Memory-Keys und -Values werden mit einem gewichteten Durchschnitt aus den aktuellen Eingaben aktualisiert.

### Funktionsweise

1.  Die Eingabesequenz wird durch die `embedding` Layer repräsentiert.
2.  Die eingebettete Eingabe wird verwendet, um den **Differentiable Memory** anzusprechen.
3.  Der Speicher liefert relevante Informationen an das RNN.
4.  Das RNN verarbeitet diese Information, um eine Ausgabe zu erzeugen.
5.  Der Memory wird mit dem Output aktualisiert.
6.  Das Modell wird mit einer `CrossEntropyLoss`-Funktion trainiert, um die Vorhersage der Zielsequenz zu optimieren.

## Installation

Um das Modell zu verwenden, stellen Sie sicher, dass Sie die folgenden Abhängigkeiten installiert haben:

```bash
pip install torch numpy tqdm scikit-learn matplotlib pyyaml optuna
```

## Verwendung

### 1. Konfiguration

Die Konfiguration des Modells erfolgt über eine `config.yaml` Datei. Hier sind die wichtigsten Parameter:

*   `data_path`: Pfad zur Eingabetextdatei.
*   `log_file`: Pfad zur Logdatei.
*   `embedding_dim`: Dimension des Embedding-Layers.
*   `memory_size`: Anzahl der Speicherplätze im differenzierbaren Speicher.
*   `learning_rate`: Lernrate für den Optimizer.
*   `batch_size`: Batch-Größe für das Training.
*   `max_seq_length`: Maximale Sequenzlänge für die Eingabe.
*   `train_size_ratio`: Verhältnis der Trainingsdaten.
*   `val_size_ratio`: Verhältnis der Validierungsdaten.
*   `epochs`: Anzahl der Trainings-Epochen.
*   `accumulation_steps`: Anzahl der Schritte zur Gradientenakkumulation.
*   `write_strength`: Stärke des Schreibens in den Memory (0-1).
*   `patience`: Geduld für Early Stopping.
*   `save_path`: Pfad zum Speichern des Modells.

Beispiel `config.yaml`:

```yaml
data_path: "data/my_text_data.txt"
log_file: "training.log"
embedding_dim: 512
memory_size: 1024
learning_rate: 0.001
batch_size: 32
max_seq_length: 128
train_size_ratio: 0.8
val_size_ratio: 0.1
epochs: 100
accumulation_steps: 4
write_strength: 0.05
patience: 10
save_path: "saved_models"
```

### 2. Training

1.  Stellen Sie sicher, dass Ihre Daten im `data_path` liegen.
2.  Starten Sie das Training mit folgendem Befehl:

```bash
python your_script_name.py
```

### 3. Modell laden und verwenden (Beispiel)
Nach dem Training kann das beste Modell geladen und verwendet werden:

```python
checkpoint = torch.load("saved_models/memory_model.pth")  # Pfad anpassen
best_config = checkpoint['config']
best_vocab = checkpoint['vocab']

# Modell neu instanziieren und Gewichte laden
best_model = Controller(
    best_config['embedding_dim'],
    best_config['embedding_dim'],
    len(best_vocab),
    best_config['memory_size'],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.eval()

# Anschließend können Sie das Modell z. B. wie folgt nutzen:
# (1) Input vorbereiten (tokenisieren, padden)
# (2) Vorhersage berechnen
# (3) Ausgabe interpretieren (decodieren)
```

### Wichtige Aspekte im Code

*   **Konfiguration:** Die `Config`-Klasse verwaltet alle wichtigen Parameter über eine YAML-Datei oder ein Dictionary.
*   **Logging:** Der Code verwendet das Python `logging`-Modul, um Trainingsdetails aufzuzeichnen.
*   **Reproduzierbarkeit:** Ein Seed wird gesetzt, um die Reproduzierbarkeit zu gewährleisten.
*   **Gerätenutzung:** Das Modell wird automatisch auf der GPU ausgeführt, falls verfügbar.
*   **Data Loaders:** Die `TextDataset` Klasse lädt und verarbeitet die Textdaten für das Training.
*   **Training:**
    *   Die `train_model_epoch`-Funktion führt eine Trainings-Epoche durch mit optionaler Gradientenakkumulation und Clipping.
    *   Die `validate_model`-Funktion evaluiert das Modell auf dem Validierungsset.
    *   `evaluate_model`-Funktion evaluiert das Modell auf dem Testset und berechnet Accuracy und F1-Score.
*   **Early Stopping:** Das Training wird vorzeitig gestoppt, wenn sich der Validierungsverlust über eine bestimmte Anzahl von Epochen nicht mehr verbessert.
*   **Speichern:** Das beste Modell und das Vokabular werden nach dem Training gespeichert.
*   **Optuna:** Die `objective`-Funktion und die `train_model`-Funktion nutzen Optuna für die Hyperparameter-Optimierung.

*   ## Hyperparameter-Optimierung

Das Modell nutzt `Optuna`, um automatisch die besten Hyperparameter zu finden. Die optimierten Parameter sind:

-   **`embedding_dim`**: Die Dimensionalität der Token-Einbettungen (128 bis 2048).
-   **`memory_size`**: Die Größe des Speichers (256 bis 4096).
-   **`learning_rate`**: Die Lernrate für den AdamW-Optimierer (1e-5 bis 1e-2, logarithmisch).
-   **`batch_size`**: Die Batch-Größe (8, 16, 32 oder 64).
-   **`accumulation_steps`**: Die Anzahl der Schritte zur Gradientenakkumulation (1 bis 10).
-   **`epochs`**: Die maximale Anzahl der Trainingsepochen (50, 100, 150 oder 200).
-   **`write_strength`**: Die Stärke, mit der der Speicher aktualisiert wird (0.01 bis 0.1, logarithmisch).

## Training mit Early Stopping und Wiederaufnahme

- Das Modell verwendet Early Stopping mit der Option, das Training wieder aufzunehmen, falls die Validierung nicht mehr besser wird.
- Wenn das Training vorzeitig abgebrochen wird, speichert das Modell den aktuellen Stand (Hyperparameter und Modell-Parameter).
- Der Optimierungsprozess mit `Optuna` startet erneut und versucht durch andere Hyperparameter die Validierungs-Performance zu erhöhen, solange bis die optimale Konfiguration gefunden ist.
- Der beste Modell-State wird während des Traings fortlaufend gespeichert und am ende des Trainings gespeichert.

## Funktionen und Klassen im Detail

### `Config` Klasse

Verwaltet Konfigurationsparameter, liest sie aus einer YAML-Datei oder einem Dictionary und validiert die Parameter.

### `DifferentiableMemory` Klasse

Implementiert den differenzierbaren Speicher.

*   `read`: Liest Werte aus dem Speicher anhand einer Query.
*   `write` und `write_as_mean`: Aktualisieren den Speicher mit neuen Keys und Values, unter Berücksichtigung des `write_strength`-Parameters und einem Mean über die Batch-Dimension.

### `Controller` Klasse

Verwaltet das Hauptmodell mit Embedding, RNN und differenzierbarem Speicher.

### `Vocabulary` Klasse

Verwaltet die Zuordnung zwischen Token und Indizes.

### `TextDataset` Klasse

Lädt, tokenisiert und präpariert die Textdaten für das Training.

### `train_model_epoch` Funktion

Trainiert das Modell für eine Epoche mit optionaler Gradientenakkumulation.

### `validate_model` Funktion

Evaluiert das Modell auf dem Validierungsset.

### `calculate_metrics` Funktion

Berechnet Genauigkeit und F1-Score.

### `evaluate_model` Funktion

Testet das Modell auf dem Testset und gibt Verlust, Genauigkeit und F1-Score zurück.

### `create_data_loaders` Funktion

Erstellt DataLoader-Objekte für das Training, die Validierung und das Testen.

### `load_and_prepare_data` Funktion

Kombiniert den Data Load Prozess.

### `create_model_and_optimizer` Funktion

Erzeugt das Model, den Optimizer, das Kriterium und den Scheduler.

### `create_vocabulary` Funktion

Erstellt die Vocabular.

### `plot_training` Funktion

Plotet den Trainings- und Validierungsverlust sowie die Testgenauigkeit und F1-Score über die Epochen hinweg.

### `setup_training` Funktion

Bereitet die notwendigen Komponenten für das Training vor (Modell, Optimizer, Daten).

### `run_training_loop` Funktion

Führt die Trainingsschleife durch und implementiert Early Stopping.

### `save_best_model` Funktion

Speichert das beste trainierte Modell, Vokabular und die Konfiguration.

### `objective` Funktion (Optuna)

Definiert die Zielfunktion für die Hyperparameteroptimierung mit Optuna.

### `train_model` Funktion (Hauptfunktion)

Steuert den gesamten Trainingsprozess, inklusive der Hyperparameteroptimierung mittels Optuna und des Trainings mit den besten Parametern.

###  `if __name__ == "__main__":` Abschnitt

Die Hauptausführungslogik des Skripts.
Hier wird die Funktion `train_model` aufgerufen, um den Trainingsprozess zu starten.
Außerdem ist ein Beispielcode zum Laden und Verwenden des Modells nach dem Training als Kommentar vorhanden.

## Nach dem Training

Nach dem Training kann das gespeicherte Modell wie folgt wiederverwendet werden (siehe Kommentar im Hauptskript):

```python
# Laden der Checkpoint-Datei
checkpoint = torch.load("memory_model.pth")
best_config = checkpoint['config']
best_vocab = checkpoint['vocab']

# Instanziierung des Modells
best_model = Controller(
    best_config['embedding_dim'],
    best_config['embedding_dim'],
    len(best_vocab),
    best_config['memory_size'],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Laden des Model-States
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.eval()
```

## Logging

Der Trainingsprozess wird detailliert in der Datei `training.log` protokolliert. Hier werden die wichtigsten Informationen gespeichert, wie z.B.:

-   Verwendete Konfiguration.
-   Verlust während des Trainings und der Validierung pro Epoche.
-   Test-Metriken (Genauigkeit und F1-Score)
-   Informationen zu vorzeitigem Abbruch.
-   Die besten Hyperparameter, welche automatisch gefunden wurden.

## Weiterentwicklung

Dieses Modell kann als Grundlage für komplexere Aufgaben dienen, z.B.:

-   Verbesserung der Architektur des Controllers.
-   Verwendung komplexerer Speichermechanismen.
-   Anwendung auf verschiedene Aufgaben, z. B. maschinelle Übersetzung oder Frage-Antwort-Systeme.

## Lizenz

Dieses Projekt ist unter der GPL v3-Lizenz lizenziert.

## Kontakt

Für Fragen oder Feedback erreichen Sie mich unter hier.



# DRLMemNet - Framework für Deep Reinforcement Learning mit speicherbasierten Netzwerkarchitekturen
# Copyright (C) 2024 Ralf Krümmel
# Projekt-Repository: https://github.com/kruemmel-python/DRLMemNet.git
#
# Dieses Programm ist freie Software: Sie können es unter den Bedingungen der
# GNU General Public License, Version 3, wie von der Free Software Foundation veröffentlicht,
# weitergeben und/oder modifizieren.
#
# Dieses Programm wird in der Hoffnung verteilt, dass es nützlich sein wird, 
# aber OHNE JEGLICHE GARANTIE; sogar ohne die implizite Garantie der 
# MARKTGÄNGIGKEIT oder EIGNUNG FÜR EINEN BESTIMMTEN ZWECK.  
# Siehe die GNU General Public License für weitere Details.
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from typing import List, Tuple, Dict, Iterator
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import yaml
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import optuna

# Debug-Modus
DEBUG_MODE = False

# Konstanten
PAD_INDEX = 0
UNK_INDEX = 1
SOS_INDEX = 2
EOS_INDEX = 3

# --------------------------------------------------
# 1) Konfigurationsklasse mit load_config
# --------------------------------------------------
@dataclass
class Config:
    data_path: str = None
    log_file: str = "training.log"
    embedding_dim: int = None
    memory_size: int = None
    learning_rate: float = None
    batch_size: int = None
    max_seq_length: int = None
    train_size_ratio: float = None
    val_size_ratio: float = None
    epochs: int = None
    accumulation_steps: int = None
    write_strength: float = None
    patience: int = None
    save_path: str = None
    device: torch.device = field(init=False)

    def __post_init__(self):
        """
        Der Konstruktor wird nach der Initialisierung aufgerufen und setzt das Gerät
        (CPU oder GPU) entsprechend der verfügbaren Hardware.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Gerät: {self.device}")

    def load_config(self, config_source: str | dict = "config.yaml"):
        """
        Lädt die Konfiguration entweder aus einer YAML-Datei (Pfad als str)
        oder direkt aus einem Dictionary, z. B. study.best_params.

        Durch strukturelles Pattern Matching können wir die Eingaben
        in unterschiedlichen Fällen behandeln.
        """
        match config_source:
            case str() as path:  # Falls der Nutzer einen Pfad (String) übergibt
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        config_yaml = yaml.safe_load(f)
                        print(f"Konfiguration aus Datei '{path}' geladen: {config_yaml}")
                        for key, value in config_yaml.items():
                            if hasattr(self, key):
                                setattr(self, key, value)
                            else:
                                logging.warning(f"Unbekannte Konfigurations-Option: {key}")
                    self._validate_config()
                except Exception as e:
                    logging.error(f"Fehler beim Laden der Konfiguration: {e}")
                    raise

            case dict() as param_dict:  # Falls ein Dictionary (z. B. study.best_params) übergeben wird
                print(f"Konfiguration aus Dictionary geladen: {param_dict}")
                for key, value in param_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        logging.warning(f"Unbekannte Konfigurations-Option: {key}")
                self._validate_config()

            case _:
                raise TypeError(
                    f"Unsupported config source type: {type(config_source).__name__}. "
                    f"Erwartet wurde str (Dateipfad) oder dict (Parameter)."
                )

    def _validate_config(self):
        """
        Stellt sicher, dass alle wichtigen Felder gesetzt und gültig sind.
        Wir brechen ab, falls eine fehlerhafte Konfiguration vorliegt.
        """
        if self.learning_rate is None or self.learning_rate <= 0:
            raise ValueError("Lernrate muss positiv sein.")
        if self.batch_size is None or self.batch_size <= 0:
            raise ValueError("Batch-Größe muss positiv sein.")
        if self.train_size_ratio is None or self.val_size_ratio is None:
            raise ValueError("Train & Val Ratio nicht definiert.")
        if not (0 < self.train_size_ratio < 1) or not (0 <= self.val_size_ratio < 1) or (self.train_size_ratio + self.val_size_ratio >= 1):
            raise ValueError("Train & Val Ratio müssen zwischen 0 und 1 liegen und kleiner als 1 sein.")

    def save_config(self) -> Dict:
        """Sichert relevante Konfigurationsattribute in einem Dict."""
        return asdict(self)

# --------------------------------------------------
# 2) Konfiguration laden (erste Datei-Ladung aus config.yaml)
# --------------------------------------------------
with open("config.yaml", "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)
CONFIG = Config(**config_data)

# Gerät
DEVICE = CONFIG.device

# Logging
logging.basicConfig(
    filename=CONFIG.log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)  # Logger

# Seed setzen für Reproduzierbarkeit
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
print("Seed gesetzt.")

# --------------------------------------------------
# 3) Klassen und Funktionen: DifferentiableMemory, Controller, Vokabular, Dataset
# --------------------------------------------------
class DifferentiableMemory(nn.Module):
    """
    Eine einfache Implementierung eines differenzierbaren Speichers.
    Enthält Keys und Values, die beim Lesen und Schreiben aktualisiert werden.
    """
    def __init__(self, memory_size: int, embedding_size: int, device=DEVICE):
        super(DifferentiableMemory, self).__init__()
        self.keys = nn.Parameter(torch.randn(memory_size, embedding_size, device=device))
        self.values = nn.Parameter(torch.randn(memory_size, embedding_size, device=device))
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.device = device
        self._initialize_weights()
        self.usage_counts = nn.Parameter(torch.ones(memory_size, device=device), requires_grad=False)

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.keys)
        nn.init.kaiming_uniform_(self.values)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query, self.keys.T)
        attention_weights = F.softmax(scores, dim=-1)
        memory_output = torch.matmul(attention_weights, self.values)

        activated_memory_indices = torch.argmax(scores, dim=-1)
        self.usage_counts[activated_memory_indices] += 1
        return memory_output

    def write(self, updated_keys: torch.Tensor, updated_values: torch.Tensor, write_strength: float):
        updated_keys = F.normalize(updated_keys, dim=-1)
        updated_values = F.normalize(updated_values, dim=-1)

        batch_size = updated_keys.shape[0]
        if batch_size != self.memory_size:
            updated_keys = updated_keys.mean(dim=0, keepdim=True).expand(self.memory_size, -1)
            updated_values = updated_values.mean(dim=0, keepdim=True).expand(self.memory_size, -1)

        with torch.no_grad():
            self.keys.copy_((1 - write_strength) * self.keys.data + write_strength * updated_keys)
            self.values.copy_((1 - write_strength) * self.values.data + write_strength * updated_values)

            self.keys.copy_(F.normalize(self.keys, dim=1))
            self.values.copy_(F.normalize(self.values, dim=1))

        if DEBUG_MODE:
            print("Aktualisierte Memory-Parameter (write).")

    def write_as_mean(self, updated_keys: torch.Tensor, updated_values: torch.Tensor, write_strength: float):
        updated_keys = F.normalize(updated_keys, dim=-1)
        updated_values = F.normalize(updated_values, dim=-1)

        with torch.no_grad():
            mean_keys = updated_keys.mean(dim=0, keepdim=True)
            mean_values = updated_values.mean(dim=0, keepdim=True)

            self.keys.copy_((1 - write_strength) * self.keys.data + write_strength * mean_keys)
            self.values.copy_((1 - write_strength) * self.values.data + write_strength * mean_values)

            self.keys.copy_(F.normalize(self.keys, dim=1))
            self.values.copy_(F.normalize(self.values, dim=1))

        if DEBUG_MODE:
            print("Aktualisierte Memory-Parameter (write_as_mean).")

class Controller(nn.Module):
    """
    Das Hauptmodell, welches die Memory-Komponente nutzt.
    """
    def __init__(self, embedding_size: int, memory_embedding_size: int, vocab_size: int, memory_size: int, device: torch.device = DEVICE):
        super(Controller, self).__init__()
        self.memory = DifferentiableMemory(memory_size, memory_embedding_size, device)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc_query = nn.Linear(embedding_size, memory_embedding_size)
        self.fc_query_act = nn.GELU()
        self.rnn = nn.GRU(memory_embedding_size, 256, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(256, 256)
        self.fc1_act = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.fc_output = nn.Linear(256, vocab_size)
        self.fc_output_act = nn.GELU()

        # Memory-Aktualisierung
        self.fc_query_memory = nn.Linear(memory_embedding_size, memory_embedding_size)
        self.fc_query_memory_act = nn.GELU()
        self.fc_value_memory = nn.Linear(vocab_size, memory_embedding_size)
        self.fc_value_memory_act = nn.GELU()

        self.device = device
        self._init_layers()

    def _init_layers(self):
        nn.init.kaiming_uniform_(self.fc_query.weight)
        nn.init.zeros_(self.fc_query.bias)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc_output.weight)
        nn.init.zeros_(self.fc_output.bias)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(inputs)
        query = self.fc_query(embedded)
        query = self.fc_query_act(query)

        memory_output = self.memory.read(query)

        rnn_output, _ = self.rnn(memory_output)
        hidden = self.fc1(rnn_output)
        hidden = self.fc1_act(hidden)
        hidden = self.dropout(hidden)

        output = self.fc_output(hidden)
        output = self.fc_output_act(output)
        return output, query

class Vocabulary:
    """
    Verwaltet die Abbildung von Token -> Index und Index -> Token.
    """
    def __init__(self, special_tokens=None):
        self.token_to_index = {}
        self.index_to_token = []
        self.special_tokens = special_tokens if special_tokens else []
        for token in self.special_tokens:
            self.add_token(token)
        self.unk_index = self.token_to_index.get("<unk>", 0)

    def add_token(self, token):
        if token not in self.token_to_index:
            self.token_to_index[token] = len(self.index_to_token)
            self.index_to_token.append(token)

    def __len__(self):
        return len(self.token_to_index)

    def __getitem__(self, token):
        return self.token_to_index.get(token, self.unk_index)

    def get_itos(self):
        return self.index_to_token

    def set_default_index(self, index):
        self.unk_index = index

    def get_index(self, token):
        return self.token_to_index.get(token, self.unk_index)

    def __contains__(self, token):
        return token in self.token_to_index

def create_vocab_from_iterator(iterator: Iterator[List[str]], special_tokens: List[str]) -> Vocabulary:
    vocab = Vocabulary(special_tokens)
    for tokens in iterator:
        for token in tokens:
            vocab.add_token(token)
    return vocab

def create_tokenizer(text: str, special_chars=r"[^a-zA-Z0-9\s.,?!]"):
    text = text.lower()
    text = re.sub(r"([.,?!])", r" \1 ", text)
    text = re.sub(special_chars, "", text)
    return re.findall(r'\b\w+|\S\b', text)

class TextDataset(Dataset):
    """
    Dataset, das Zeilen aus einer Textdatei lädt, tokenisiert und in Zahlen übersetzt.
    """
    def __init__(self, data_path: str, tokenizer, vocab: Vocabulary, max_seq_length: int):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.data_path = data_path
        self.lines = self._load_lines(data_path)

    def _load_lines(self, data_path: str) -> List[str]:
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Fehler beim Einlesen der Datei {data_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.lines[idx]
        tokens = self.tokenizer(text)
        tokens_ids = [self.vocab[token] for token in tokens]

        # Eingabe (Input)
        input_ids = [self.vocab["<sos>"]] + tokens_ids
        input_ids = input_ids[:self.max_seq_length]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids = F.pad(input_ids, (0, padding_length), value=self.vocab["<pad>"])

        # Ziel (Target)
        target_ids = tokens_ids + [self.vocab["<eos>"]]
        target_ids = target_ids[:self.max_seq_length]
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        target_padding_length = self.max_seq_length - len(target_ids)
        target_ids = F.pad(target_ids, (0, target_padding_length), value=self.vocab["<pad>"])

        return input_ids, target_ids

# --------------------------------------------------
# 4) Trainings- und Evaluierungsfunktionen
# --------------------------------------------------
def train_model_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion,
    epoch: int,
    accumulation_steps: int,
    write_strength: float,
    clip_value: float
) -> float:
    """
    Führt eine komplette Trainings-Epoche durch mit optionaler Gradientenakkumulation.
    """
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoche {epoch + 1:03d} (Training)")

    for batch_idx, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs, query = model(inputs)
        loss = criterion(F.log_softmax(outputs, dim=-1).view(-1, outputs.size(-1)), targets.view(-1))
        loss = loss / accumulation_steps

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()

            # Memory-Update
            query_for_memory = model.fc_query_memory(query)
            query_for_memory = model.fc_query_memory_act(query_for_memory)
            outputs_for_memory = model.fc_value_memory(outputs)
            outputs_for_memory = model.fc_value_memory_act(outputs_for_memory)

            query_for_memory = query_for_memory.mean(dim=1)
            outputs_for_memory = outputs_for_memory.mean(dim=1)

            model.memory.write_as_mean(query_for_memory, outputs_for_memory, write_strength)

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    logger.info(f"Epoche {epoch + 1:03d} Trainingsverlust: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def validate_model(model: nn.Module, val_dataloader: DataLoader, criterion) -> float:
    """
    Validiert das Modell und gibt den durchschnittlichen Validierungsverlust zurück.
    """
    if len(val_dataloader) == 0:
        print("Keine Validierungsdaten verfügbar. Überspringe Validierung.")
        logger.warning("Keine Validierungsdaten verfügbar. Überspringe Validierung.")
        return float('inf')

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_dataloader:
            val_inputs, val_targets = val_inputs.to(DEVICE), val_targets.to(DEVICE)
            val_outputs, _ = model(val_inputs)
            val_loss += criterion(
                F.log_softmax(val_outputs, dim=-1).view(-1, val_outputs.size(-1)),
                val_targets.view(-1)
            ).item()

    avg_val_loss = val_loss / len(val_dataloader)
    logger.info(f'Validierungsverlust: {avg_val_loss:.4f}')
    return avg_val_loss

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, pad_index: int) -> Tuple[float, float]:
    """
    Berechnet Accuracy und F1-Score und ignoriert dabei Padding-Tokens.
    """
    predictions = torch.argmax(predictions, dim=-1)
    mask = targets != pad_index
    if not mask.any():
        return 0.0, 0.0

    masked_targets = targets[mask].cpu().numpy()
    masked_predictions = predictions[mask].cpu().numpy()

    if DEBUG_MODE:
        print("Maskierte Ziele (erste 10):", masked_targets[:10])
        print("Maskierte Vorhersagen (erste 10):", masked_predictions[:10])

    accuracy = accuracy_score(masked_targets, masked_predictions)
    f1 = f1_score(masked_targets, masked_predictions, average='weighted', zero_division=1)
    return accuracy, f1

def evaluate_model(model: nn.Module, test_dataloader: DataLoader, criterion, vocab) -> Tuple[float, float, float]:
    """
    Testet das Modell auf dem Test-Dataset und berechnet den Durchschnittsverlust,
    die Genauigkeit und den F1-Score. Speicherschonende Variante, die batchweise akkumuliert.
    """
    model.eval()
    total_batches = len(test_dataloader)

    if total_batches == 0:
        print("Kein Test-Daten verfügbar.")
        logger.warning("Kein Test-Daten verfügbar.")
        return float('inf'), 0.0, 0.0

    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (test_inputs, test_targets) in enumerate(tqdm(test_dataloader, desc="Evaluierung")):
            test_inputs, test_targets = test_inputs.to(DEVICE), test_targets.to(DEVICE)
            test_outputs, _ = model(test_inputs)

            # Loss
            loss = criterion(
                F.log_softmax(test_outputs, dim=-1).view(-1, test_outputs.size(-1)),
                test_targets.view(-1)
            )
            total_loss += loss.item()

            # Batch-Accuracy und Batch-F1
            batch_accuracy, batch_f1 = calculate_metrics(test_outputs, test_targets, vocab.get_index("<pad>"))
            batch_size = test_inputs.size(0)

            total_accuracy += batch_accuracy * batch_size
            total_f1 += batch_f1 * batch_size
            total_samples += batch_size

    avg_test_loss = total_loss / total_batches
    avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0

    logger.info(f"Testverlust: {avg_test_loss:.4f}, Testgenauigkeit: {avg_accuracy:.4f}, Test F1-Score: {avg_f1:.4f}")
    print(f"Testverlust: {avg_test_loss:.4f}, Testgenauigkeit: {avg_accuracy:.4f}, Test F1-Score: {avg_f1:.4f}")

    return avg_test_loss, avg_accuracy, avg_f1

def create_data_loaders(dataset, batch_size, train_size_ratio, val_size_ratio):
    total_size = len(dataset)
    train_size = int(train_size_ratio * total_size)
    val_size = max(1, int(val_size_ratio * total_size))
    test_size = total_size - train_size - val_size

    if test_size < 1:
        test_size = 1
        train_size -= 1
    if val_size < 1:
        val_size = 1
        train_size -= 1

    logger.info(f"Datensätze Größen: train_size={train_size}, val_size={val_size}, test_size={test_size}")

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def load_and_prepare_data(config: Config, tokenizer, vocab):
    dataset = TextDataset(config.data_path, tokenizer, vocab, config.max_seq_length)
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
        dataset,
        config.batch_size,
        config.train_size_ratio,
        config.val_size_ratio
    )
    return train_dataloader, val_dataloader, test_dataloader

def create_model_and_optimizer(config: Config, vocab_len):
    model = Controller(
        config.embedding_dim,
        config.embedding_dim,
        vocab_len,
        config.memory_size,
        DEVICE
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    return model, optimizer, criterion, scheduler

def create_vocabulary(config: Config):
    tokenizer = lambda text: create_tokenizer(text)
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

    def yield_tokens(file_path: str) -> Iterator[List[str]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield tokenizer(line.strip())
        except Exception as e:
            logger.error(f"Fehler beim Einlesen der Datei: {e}")
            raise

    vocab = create_vocab_from_iterator(
        yield_tokens(config.data_path),
        special_tokens
    )
    vocab.set_default_index(vocab["<unk>"])
    return tokenizer, vocab

def plot_training(train_losses, val_losses, test_accuracies, test_f1_scores):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Trainingsverlust')
    plt.plot(val_losses, label='Validierungsverlust')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.title('Trainings- und Validierungsverlust')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Testgenauigkeit')
    plt.plot(test_f1_scores, label='Test F1-Score')
    plt.xlabel('Epoche')
    plt.ylabel('Metrik')
    plt.legend()
    plt.title('Testgenauigkeit und F1-Score')

    plt.tight_layout()
    plt.show()

def setup_training(config: Config, tokenizer, vocab):
    model, optimizer, criterion, scheduler = create_model_and_optimizer(config, len(vocab))
    train_dataloader, val_dataloader, test_dataloader = load_and_prepare_data(config, tokenizer, vocab)
    return model, optimizer, criterion, scheduler, train_dataloader, val_dataloader, test_dataloader

def run_training_loop(
    model: nn.Module,
    config: Config,
    optimizer: optim.Optimizer,
    criterion,
    scheduler: optim.lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    vocab
) -> Tuple[List[float], List[float], List[float], List[float]]:
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config.patience
    save_path = Path(config.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []
    test_accuracies = []
    test_f1_scores = []

    for epoch in range(config.epochs):
        try:
            logger.info(f"Epoche {epoch + 1} gestartet.")
            train_loss = train_model_epoch(
                model,
                train_dataloader,
                optimizer,
                criterion,
                epoch,
                accumulation_steps=config.accumulation_steps,
                write_strength=config.write_strength,
                clip_value=1.0
            )

            val_loss = validate_model(model, val_dataloader, criterion)
            if val_loss == float('inf'):
                logger.warning("Validierung wurde übersprungen.")
            else:
                scheduler.step()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Wir merken uns den aktuellen State
                    best_model_state = model.state_dict()
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), save_path / f"memory_model_epoch_{epoch + 1}.pth")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve > patience:
                    logger.info("Vorzeitiger Abbruch ausgelöst (Early Stopping).")
                    print("Training wird vorzeitig abgebrochen.")
                    break

            # Neue, speicherschonende Evaluierung
            test_loss, accuracy, f1 = evaluate_model(model, test_dataloader, criterion, vocab)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_accuracies.append(accuracy)
            test_f1_scores.append(f1)

        except Exception as e:
            logger.error(f"Fehler in Epoche {epoch}: {e}")
            print(f"Fehler in Epoche {epoch}: {e}")
            break

    # Letzte Evaluation nach dem Training
    test_loss, accuracy, f1 = evaluate_model(model, test_dataloader, criterion, vocab)
    return train_losses, val_losses, test_accuracies, test_f1_scores

def save_best_model(
    config: Config,
    model: nn.Module,
    vocab,
    train_losses: List[float],
    val_losses: List[float],
    test_accuracies: List[float],
    test_f1_scores: List[float]
):
    config_to_save = config.save_config()
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': config_to_save
    }, Path(config.save_path) / "memory_model.pth")
    logger.info("Bestes Modell und Vokabular erfolgreich gespeichert!")
    print("Bestes Modell erfolgreich gespeichert.")

# --------------------------------------------------
# 5) Optuna-Objective und Haupt-Trainingsfunktion
# --------------------------------------------------
def objective(trial: optuna.Trial, config: Config) -> float:
    # Hyperparameter definieren
    # (Ab Optuna 3.0: prefer suggest_float(..., log=True) anstelle von suggest_loguniform)
    config.embedding_dim = trial.suggest_int('embedding_dim', 128, 2048, step=128)
    config.memory_size = trial.suggest_int('memory_size', 256, 4096, step=256)
    config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    config.accumulation_steps = trial.suggest_int('accumulation_steps', 1, 10)
    config.epochs = trial.suggest_int('epochs', 50, 200, step=50)
    config.write_strength = trial.suggest_float('write_strength', 0.01, 0.1, log=True)

    # Setup
    tokenizer, vocab = create_vocabulary(config)
    model, optimizer, criterion, scheduler, train_dataloader, val_dataloader, test_dataloader = setup_training(config, tokenizer, vocab)

    # Trainingsloop
    train_losses, val_losses, test_accuracies, test_f1_scores = run_training_loop(
        model,
        config,
        optimizer,
        criterion,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        vocab
    )

    best_val_loss = min(val_losses) if len(val_losses) > 0 else float('inf')
    return best_val_loss

def train_model(config: Config):
    print("train_model() gestartet.")
    logger.info("train_model() gestartet.")

    # Optuna-Studie erstellen
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, config), n_trials=50, n_jobs=1, timeout=3600)

    # Beste Hyperparameter
    print("Beste Hyperparameter: ", study.best_params)
    logger.info(f"Beste Hyperparameter: {study.best_params}")

    # Übernahme der besten Parameter ins Config-Objekt
    config.load_config(study.best_params)

    # Finale Trainingsschleife mit den besten Parametern
    tokenizer, vocab = create_vocabulary(config)
    model, optimizer, criterion, scheduler, train_dataloader, val_dataloader, test_dataloader = setup_training(config, tokenizer, vocab)

    train_losses, val_losses, test_accuracies, test_f1_scores = run_training_loop(
        model,
        config,
        optimizer,
        criterion,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        vocab
    )

    # Speichern des besten Modells
    save_best_model(config, model, vocab, train_losses, val_losses, test_accuracies, test_f1_scores)
    # Visualisierung
    plot_training(train_losses, val_losses, test_accuracies, test_f1_scores)

    print("train_model() abgeschlossen.")
    logger.info("train_model() abgeschlossen.")

# --------------------------------------------------
# 6) Hauptausführung
# --------------------------------------------------
if __name__ == "__main__":
    # Startet den Trainingsprozess
    train_model(CONFIG)

    # --- Beispiel: Trainiertes Modell laden und verwenden ---
    """
    Falls Sie nach dem Training mit den besten Parametern Ihr Modell
    in einer echten Anwendung nutzen möchten, denken Sie daran,
    es samt Vokabular und Config zu laden.

    Beispiel:
    --------
    checkpoint = torch.load("memory_model.pth")  # Pfad anpassen, falls nötig
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
    # (1) Input vorbereiten
    # (2) Vorhersage berechnen
    # (3) Ausgabe interpretieren
    """

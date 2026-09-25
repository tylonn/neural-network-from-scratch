"""Rede neural manual para classificação de círculos concêntricos.

Projeto baseado no exemplo4.py da aula de 21/08/2026, mas com:
- nova base de dados: sklearn.datasets.make_circles;
- arquitetura fixa 2 -> 4 -> 3 -> 1;
- tanh nas camadas ocultas e sigmoid na saída;
- entropia cruzada binária como função de custo;
- forward e backpropagation implementados manualmente com NumPy;
- comparação opcional com uma rede equivalente em Keras/TensorFlow.

A implementação manual não usa uma biblioteca de redes neurais para o treinamento.
O scikit-learn é usado apenas para gerar/separar a base e o matplotlib para gráficos.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


Array = np.ndarray
Params = Dict[str, Array]

SEED = 42


def sigmoid(z: Array) -> Array:
    """Sigmoide numericamente estável para a camada de saída."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y: Array, y_hat: Array) -> float:
    """Entropia cruzada binária média."""
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))


def accuracy(y: Array, y_hat: Array) -> float:
    """Acurácia usando limiar de 0.5 na probabilidade prevista."""
    pred = (y_hat >= 0.5).astype(int)
    return float(np.mean(pred == y.astype(int)))


def make_dataset(
    n_samples: int = 300,
    noise: float = 0.08,
    factor: float = 0.45,
    test_size: float = 0.25,
    seed: int = SEED,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Cria uma base pequena de classificação binária com círculos concêntricos."""
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=seed,
    )
    y = y.reshape(-1, 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X, y, X_train, X_test, y_train, y_test


def initialize_parameters(seed: int = SEED) -> Params:
    """Inicializa os parâmetros da arquitetura FIXA 2 -> 4 -> 3 -> 1.

    As escalas seguem a ideia de Xavier/Glorot para evitar ativações iniciais
    excessivamente grandes quando usamos tanh.
    """
    rng = np.random.default_rng(seed)

    return {
        "W1": rng.normal(0.0, np.sqrt(1.0 / 2.0), size=(2, 4)),
        "b1": np.zeros((1, 4)),
        "W2": rng.normal(0.0, np.sqrt(1.0 / 4.0), size=(4, 3)),
        "b2": np.zeros((1, 3)),
        "W3": rng.normal(0.0, np.sqrt(1.0 / 3.0), size=(3, 1)),
        "b3": np.zeros((1, 1)),
    }


def forward(X: Array, p: Params) -> Dict[str, Array]:
    """Executa o forward pass completo da rede manual.

    Camada 1:
        z1 = X W1 + b1
        a1 = tanh(z1)

    Camada 2:
        z2 = a1 W2 + b2
        a2 = tanh(z2)

    Saída:
        z3 = a2 W3 + b3
        y_hat = sigmoid(z3)
    """
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.tanh(z1)

    z2 = a1 @ p["W2"] + p["b2"]
    a2 = np.tanh(z2)

    z3 = a2 @ p["W3"] + p["b3"]
    y_hat = sigmoid(z3)

    return {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2,
        "z3": z3,
        "y_hat": y_hat,
    }


def backward(X: Array, y: Array, cache: Dict[str, Array], p: Params) -> Params:
    """Calcula manualmente o backpropagation da arquitetura 2 -> 4 -> 3 -> 1.

    Para BCE + sigmoid, a derivada em relação a z3 simplifica para:
        dz3 = (y_hat - y) / m

    Para tanh:
        d/dz tanh(z) = 1 - tanh(z)^2
    """
    m = X.shape[0]
    a1 = cache["a1"]
    a2 = cache["a2"]
    y_hat = cache["y_hat"]

    # Camada de saída -------------------------------------------------------
    dz3 = (y_hat - y) / m
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)

    # Segunda camada oculta ------------------------------------------------
    da2 = dz3 @ p["W3"].T
    dz2 = da2 * (1.0 - a2**2)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Primeira camada oculta -----------------------------------------------
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (1.0 - a1**2)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3,
    }


def update_parameters(p: Params, grads: Params, learning_rate: float) -> None:
    """Aplica uma etapa de gradiente descendente in-place."""
    for name in p:
        p[name] -= learning_rate * grads[name]


def train_manual(
    X_train: Array,
    y_train: Array,
    epochs: int = 2500,
    learning_rate: float = 0.1,
    seed: int = SEED,
    print_every: int = 250,
) -> Tuple[Params, list[float]]:
    """Treina a rede manual por gradiente descendente em lote completo."""
    p = initialize_parameters(seed)
    history: list[float] = []

    initial = forward(X_train, p)["y_hat"]
    print(f"Acurácia manual antes do treinamento: {accuracy(y_train, initial):.2%}")

    for epoch in range(epochs + 1):
        cache = forward(X_train, p)
        loss = binary_cross_entropy(y_train, cache["y_hat"])
        history.append(loss)

        if epoch % print_every == 0 or epoch == epochs:
            train_acc = accuracy(y_train, cache["y_hat"])
            print(f"Época {epoch:4d} | loss={loss:.6f} | acc_treino={train_acc:.2%}")

        if epoch == epochs:
            break

        grads = backward(X_train, y_train, cache, p)
        update_parameters(p, grads, learning_rate)

    return p, history


def numerical_gradient_check(
    X: Array,
    y: Array,
    p: Params,
    epsilon: float = 1e-5,
    checks_per_tensor: int = 3,
    seed: int = SEED,
) -> float:
    """Compara derivadas analíticas com diferenças finitas em alguns parâmetros.

    Não participa do treinamento. É apenas uma verificação adicional de que o
    backpropagation manual foi implementado de forma coerente.
    """
    rng = np.random.default_rng(seed)
    cache = forward(X, p)
    analytic = backward(X, y, cache, p)
    relative_errors: list[float] = []

    for name, tensor in p.items():
        all_indices = list(np.ndindex(tensor.shape))
        count = min(checks_per_tensor, len(all_indices))
        chosen = rng.choice(len(all_indices), size=count, replace=False)

        for pos in chosen:
            idx = all_indices[int(pos)]
            original = tensor[idx]

            tensor[idx] = original + epsilon
            loss_plus = binary_cross_entropy(y, forward(X, p)["y_hat"])

            tensor[idx] = original - epsilon
            loss_minus = binary_cross_entropy(y, forward(X, p)["y_hat"])

            tensor[idx] = original
            numeric = (loss_plus - loss_minus) / (2.0 * epsilon)
            analytic_value = analytic[name][idx]

            denom = max(1e-12, abs(numeric) + abs(analytic_value))
            relative_errors.append(abs(numeric - analytic_value) / denom)

    return float(max(relative_errors))


def predict_proba(X: Array, p: Params) -> Array:
    """Retorna a probabilidade prevista para cada amostra."""
    return forward(X, p)["y_hat"]


def predict_proba_batched(
    X: Array,
    p: Params,
    batch_size: int = 128,
) -> Array:
    """Calcula probabilidades em lotes menores.

    A função é usada principalmente na geração da fronteira de decisão.
    O gráfico avalia mais de cem mil pontos; dividir essa avaliação em lotes
    pequenos evita operações matriciais grandes durante a visualização sem
    alterar a rede, os pesos treinados ou as previsões.
    """
    outputs: list[Array] = []

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        outputs.append(predict_proba(X[start:end], p))

    return np.vstack(outputs)


def plot_dataset(X: Array, y: Array, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolors="k", alpha=0.8)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Base de dados: círculos concêntricos")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_loss(history: list[float], output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(history)
    plt.xlabel("Época")
    plt.ylabel("Entropia cruzada binária")
    plt.title("Curva de treinamento da rede manual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_decision_boundary(
    X: Array,
    y: Array,
    p: Params,
    output_path: Path,
    title: str = "Fronteira de decisão da rede manual",
) -> None:
    margin = 0.35
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 350),
        np.linspace(y_min, y_max, 350),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Avalia a malha em lotes para evitar uma única multiplicação matricial
    # muito grande durante a geração do gráfico.
    zz = predict_proba_batched(grid, p).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, zz, levels=np.linspace(0, 1, 21), cmap="coolwarm", alpha=0.45)
    plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolors="k", alpha=0.85)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_architecture(output_path: Path) -> None:
    """Gera a figura da arquitetura 2 -> 4 -> 3 -> 1."""
    layers = [
        (0.0, 2, "Entrada\n2 atributos"),
        (1.7, 4, "Oculta 1\n4 neurônios\ntanh"),
        (3.4, 3, "Oculta 2\n3 neurônios\ntanh"),
        (5.1, 1, "Saída\n1 neurônio\nsigmoid"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    positions: list[list[tuple[float, float]]] = []

    for x, count, label in layers:
        ys = np.linspace(-(count - 1) / 2, (count - 1) / 2, count)
        current = []
        for y in ys:
            circle = plt.Circle((x, y), 0.18, fill=False, linewidth=2)
            ax.add_patch(circle)
            current.append((x, y))
        positions.append(current)
        ax.text(x, -2.15, label, ha="center", va="top", fontsize=11)

    for left, right in zip(positions[:-1], positions[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                ax.plot([x1 + 0.18, x2 - 0.18], [y1, y2], linewidth=0.8, alpha=0.55)

    ax.set_xlim(-0.6, 5.7)
    ax.set_ylim(-2.7, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Arquitetura da rede neural manual: 2 → 4 → 3 → 1", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_computational_graph(output_path: Path) -> None:
    """Gera um resumo visual do forward e do backward completos."""
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis("off")

    forward_text = (
        "FORWARD\n\n"
        "z₁ = XW₁ + b₁\n"
        "a₁ = tanh(z₁)\n\n"
        "z₂ = a₁W₂ + b₂\n"
        "a₂ = tanh(z₂)\n\n"
        "z₃ = a₂W₃ + b₃\n"
        "ŷ = sigmoid(z₃)\n\n"
        "L = BCE(y, ŷ)"
    )

    backward_text = (
        "BACKWARD\n\n"
        "dz₃ = (ŷ - y) / m\n"
        "dW₃ = a₂ᵀ dz₃\n"
        "db₃ = Σ dz₃\n\n"
        "da₂ = dz₃ W₃ᵀ\n"
        "dz₂ = da₂ ⊙ (1 - a₂²)\n"
        "dW₂ = a₁ᵀ dz₂\n"
        "db₂ = Σ dz₂\n\n"
        "da₁ = dz₂ W₂ᵀ\n"
        "dz₁ = da₁ ⊙ (1 - a₁²)\n"
        "dW₁ = Xᵀ dz₁\n"
        "db₁ = Σ dz₁"
    )

    update_text = (
        "ATUALIZAÇÃO\n\n"
        "Wₖ ← Wₖ - η dWₖ\n"
        "bₖ ← bₖ - η dbₖ\n\n"
        "k ∈ {1, 2, 3}"
    )

    box = dict(boxstyle="round,pad=0.8", fc="white", ec="black", lw=1.5)
    ax.text(0.05, 0.53, forward_text, transform=ax.transAxes, va="center", fontsize=13, bbox=box)
    ax.text(0.42, 0.53, backward_text, transform=ax.transAxes, va="center", fontsize=13, bbox=box)
    ax.text(0.80, 0.53, update_text, transform=ax.transAxes, va="center", fontsize=13, bbox=box)

    ax.annotate("", xy=(0.40, 0.53), xytext=(0.29, 0.53), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.78, 0.53), xytext=(0.69, 0.53), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("nova época", xy=(0.14, 0.18), xytext=(0.87, 0.18), xycoords="axes fraction",
                ha="center", va="center", arrowprops=dict(arrowstyle="->", lw=1.8,
                connectionstyle="arc3,rad=-0.25"))

    ax.set_title("Gráfico computacional resumido: forward, backpropagation e atualização", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_keras_reference(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict | None:
    """Treina a mesma arquitetura em Keras quando TensorFlow estiver instalado."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ModuleNotFoundError:
        print("\nKeras/TensorFlow não está instalado. Comparação automática ignorada.")
        print("Instale as dependências de requirements.txt para executar essa etapa.")
        return None

    keras.utils.set_random_seed(seed)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(4, activation="tanh"),
            keras.layers.Dense(3, activation="tanh"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=len(X_train),
        verbose=0,
        shuffle=False,
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }


def write_results(
    output_path: Path,
    manual_train_loss: float,
    manual_train_acc: float,
    manual_test_loss: float,
    manual_test_acc: float,
    gradient_error: float,
    keras_results: dict | None,
    epochs: int,
    learning_rate: float,
) -> None:
    lines = [
        "RESULTADOS DO PROJETO",
        "=====================",
        "",
        "Arquitetura manual: 2 -> 4 -> 3 -> 1",
        "Ativações: tanh -> tanh -> sigmoid",
        "Loss: binary cross-entropy",
        f"Épocas: {epochs}",
        f"Taxa de aprendizado: {learning_rate}",
        "",
        "Rede manual (NumPy):",
        f"  loss treino: {manual_train_loss:.6f}",
        f"  acurácia treino: {manual_train_acc:.2%}",
        f"  loss teste: {manual_test_loss:.6f}",
        f"  acurácia teste: {manual_test_acc:.2%}",
        f"  maior erro relativo no gradient check: {gradient_error:.3e}",
        "",
    ]

    if keras_results is None:
        lines += [
            "Rede Keras: não executada (TensorFlow ausente no ambiente atual).",
            "Execute novamente após instalar requirements.txt.",
        ]
    else:
        lines += [
            "Rede equivalente em Keras:",
            f"  loss treino: {keras_results['train_loss']:.6f}",
            f"  acurácia treino: {keras_results['train_acc']:.2%}",
            f"  loss teste: {keras_results['test_loss']:.6f}",
            f"  acurácia teste: {keras_results['test_acc']:.2%}",
        ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rede neural manual 2-4-3-1 para make_circles")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--skip-keras", action="store_true", help="Não executa a comparação com Keras")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(__file__).resolve().parent / "resultados"
    output_dir.mkdir(exist_ok=True)

    X, y, X_train, X_test, y_train, y_test = make_dataset(n_samples=args.samples)

    plot_dataset(X, y, output_dir / "dataset.png")
    plot_architecture(output_dir / "arquitetura.png")
    plot_computational_graph(output_dir / "grafo_computacional.png")

    print("Rede manual: 2 entradas -> 4 tanh -> 3 tanh -> 1 sigmoid")
    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    print(f"Taxa de aprendizado: {args.learning_rate} | Épocas: {args.epochs}\n")

    initial_params = initialize_parameters(SEED)
    gradient_error = numerical_gradient_check(
        X_train[:8], y_train[:8], initial_params, checks_per_tensor=3
    )
    print(f"Gradient check - maior erro relativo: {gradient_error:.3e}\n")

    params, history = train_manual(
        X_train,
        y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=SEED,
    )

    train_hat = predict_proba(X_train, params)
    test_hat = predict_proba(X_test, params)
    manual_train_loss = binary_cross_entropy(y_train, train_hat)
    manual_test_loss = binary_cross_entropy(y_test, test_hat)
    manual_train_acc = accuracy(y_train, train_hat)
    manual_test_acc = accuracy(y_test, test_hat)

    print("\nResultado da rede manual:")
    print(f"  treino: loss={manual_train_loss:.6f} | acc={manual_train_acc:.2%}")
    print(f"  teste : loss={manual_test_loss:.6f} | acc={manual_test_acc:.2%}")

    plot_loss(history, output_dir / "curva_loss.png")
    plot_decision_boundary(X, y, params, output_dir / "fronteira_decisao.png")

    keras_results = None
    if not args.skip_keras:
        keras_results = train_keras_reference(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=SEED,
        )
        if keras_results is not None:
            print("\nResultado da rede equivalente em Keras:")
            print(
                f"  treino: loss={keras_results['train_loss']:.6f} | "
                f"acc={keras_results['train_acc']:.2%}"
            )
            print(
                f"  teste : loss={keras_results['test_loss']:.6f} | "
                f"acc={keras_results['test_acc']:.2%}"
            )

    write_results(
        output_dir / "resultados.txt",
        manual_train_loss,
        manual_train_acc,
        manual_test_loss,
        manual_test_acc,
        gradient_error,
        keras_results,
        args.epochs,
        args.learning_rate,
    )

    print(f"\nArquivos gerados em: {output_dir}")


if __name__ == "__main__":
    main()

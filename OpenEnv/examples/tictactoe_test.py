"""
Quick Tic-Tac-Toe test using opponent_model_loader.load_opponent_model.

- Uses a small model for a fast smoke test.
- Prompts the model to return a single move index (0-8). If the model output can't be parsed
  into a valid move, the harness falls back to a random legal move.
- Prints the board after each move and reports the result.

"""
from pathlib import Path
import random
import re
import sys
from typing import List, Optional, Tuple
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "src" / "envs" / "openspiel_env" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from opponent_model_loader import load_opponent_model

class TicTacToe:
    def __init__(self):
        # board positions 0..8, empty = ' '
        self.board = [" "] * 9

    def play(self, pos: int, symbol: str) -> bool:
        if self.board[pos] != " ":
            return False
        self.board[pos] = symbol
        return True

    def legal_moves(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == " "]

    def winner(self) -> Optional[str]:
        b = self.board
        wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]
        for a, c, d in wins:
            if b[a] == b[c] == b[d] != " ":
                return b[a]
        if " " not in b:
            return "draw"
        return None

    def render(self) -> str:
        b = self.board
        lines = []
        for i in range(0, 9, 3):
            lines.append(" | ".join(b[i : i + 3]))
        return "\n---------\n".join(lines)


def build_prompt_for_move(board: List[str], my_symbol: str, opp_symbol: str) -> str:
    # Simple textual board representation; ask model to pick a single index 0-8
    rows = []
    for i in range(0, 9, 3):
        rows.append(",".join(board[i : i + 3]))
    board_text = ";".join(rows)
    prompt = (
        f"TicTacToe board (rows semicolon-separated). Empty cells are ' '.\n"
        f"Board: {board_text}\n"
        f"You are playing as '{my_symbol}'. Opponent is '{opp_symbol}'.\n"
        "Choose one move and output ONLY the index (0-8) of the cell you want to play.\n"
        "Indices mapping:\n"
        "0 1 2\n3 4 5\n6 7 8\n\n"
        "Output format example: 4\n"
    )
    return prompt


def parse_move_from_text(text: str) -> Optional[int]:
    # Find first digit 0-8
    m = re.search(r"[0-8]", text)
    if not m:
        return None
    return int(m.group(0))


def get_model_move(model, tokenizer, board: List[str], my_symbol: str, opp_symbol: str) -> int:
    prompt = build_prompt_for_move(board, my_symbol, opp_symbol)
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        # Move tensors to model device if model has .device attribute for convenience
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass

    try:
        out = model.generate(**inputs, max_new_tokens=16, do_sample=True, top_k=50, temperature=0.8)
        gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        # Some model wrappers (e.g. FastLanguageModel) may expose different .generate interfaces.
        # Try a couple of fallbacks:
        try:
            gen_text = model.generate_text(prompt, max_new_tokens=16)  # example alt API
        except Exception:
            gen_text = ""
    # Only keep the generated suffix (after the prompt), if any
    if gen_text.startswith(prompt):
        generated_suffix = gen_text[len(prompt) :].strip()
    else:
        generated_suffix = gen_text.strip()

    move = parse_move_from_text(generated_suffix)
    return -1 if move is None else move


def main():
    model_id = "unsloth/Llama-3.2-3B"
    peft_kwargs = None  #
    print("Loading opponent1 (unsloth/Llama-3.2-3B) ...")
    opponent1_model, opponent1_tokenizer = load_opponent_model(
        "opponent1", model_id, load_in_4bit=False, peft_kwargs=peft_kwargs
    )
    print("Loaded opponent1.")

    print("Loading opponent2 (unsloth/Llama-3.2-3B) ...")
    opponent2_model, opponent2_tokenizer = load_opponent_model(
        "opponent2", model_id, load_in_4bit=False, peft_kwargs=peft_kwargs
    )
    print("Loaded opponent2.")

    game = TicTacToe()
    # Assign symbols
    p1_sym = "X"
    p2_sym = "O"
    current = 1
    turn = 0

    while True:
        turn += 1
        if current == 1:
            model, tokenizer, sym, opp_sym = opponent1_model, opponent1_tokenizer, p1_sym, p2_sym
        else:
            model, tokenizer, sym, opp_sym = opponent2_model, opponent2_tokenizer, p2_sym, p1_sym

        legal = game.legal_moves()
        if not legal:
            print("No legal moves left.")
            break

        move = get_model_move(model, tokenizer, game.board, sym, opp_sym)
        if move == -1 or move not in legal:
            # fallback: random legal move
            move = random.choice(legal)

        ok = game.play(move, sym)
        if not ok:
            # should not happen because we check legality, but guarded anyway
            print(f"Illegal play attempted at {move} by player {current}. Choosing random legal move.")
            move = random.choice(legal)
            game.play(move, sym)

        print(f"\nTurn {turn}: Player {current} ({sym}) played {move}")
        print(game.render())

        w = game.winner()
        if w:
            if w == "draw":
                print("\nGame result: Draw")
            else:
                print(f"\nGame result: {w} wins")
            break

        current = 2 if current == 1 else 1

    print("\nTest finished.")


if __name__ == "__main__":
    main()
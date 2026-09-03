import random
from pathlib import Path

from reasoning_core.tasks.generated.wave5.s54_polynomial_gcd.s54_polynomial_gcd import PolynomialGcd


def main():
    random.seed(4028055723)
    out = Path(__file__).with_name("samples_S54.md")
    lines = []
    lines.append("# Polynomial Gcd samples\n")
    for level in (0, 2, 5):
        t = PolynomialGcd()
        t.config.set_level(level)
        lines.append(f"## Level {level}\n")
        for _ in range(2):
            e = t.generate_example()
            prompt = t.render_prompt(e.metadata)
            lines.append("### Example\n")
            lines.append(prompt)
            lines.append("\n**Answer:**\n")
            lines.append(e.answer)
            lines.append("\n")
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

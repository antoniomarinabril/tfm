#!/usr/bin/env python3
"""
Agente LLM para redactar contenido académico en español a partir de papers científicos.
Usa la API de Claude (Anthropic) para generar texto original que capture las ideas
principales del paper, evitando el plagio mediante paráfrasis y reestructuración.

Uso:
    export ANTHROPIC_API_KEY="tu-api-key"
    python3 thesis_writer_agent.py --pdf gnn_tfm.pdf --output redaccion_tfm.md

Opciones:
    --pdf       Ruta al archivo PDF del paper
    --output    Ruta del archivo de salida (por defecto: redaccion_tfm.md)
    --model     Modelo de Claude a usar (por defecto: claude-sonnet-4-20250514)
"""

import argparse
import os
import sys
import json
from pathlib import Path

from PyPDF2 import PdfReader
import anthropic


# ── Configuración ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Eres un asistente académico experto en redacción de trabajos fin de máster (TFM) \
en español. Tu tarea es leer un paper científico y redactar el contenido con tus \
propias palabras en español, de forma que:

1. **No sea una traducción literal** del paper original. Debes parafrasear, \
reestructurar y reformular las ideas.
2. **Mantenga el rigor académico**: usa terminología técnica correcta, cita \
conceptos cuando sea necesario, y mantén un tono formal.
3. **Capture todas las ideas principales**: motivación, problema, contribuciones, \
metodología, resultados y conclusiones.
4. **Use un estilo propio y natural en español**: no calcos del inglés, frases \
bien construidas, conectores lógicos adecuados.
5. **Incluya referencias al paper original** cuando cites datos o resultados \
específicos (ej: "Según Egressy et al. (2024)...").
6. **Estructura el texto en secciones claras** con encabezados en markdown.

IMPORTANTE: Tu objetivo es que el texto resultante sea original y no detectable \
como plagio, pero que transmita fielmente el contenido científico del paper.
"""


# ── Extracción de texto del PDF ────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae todo el texto de un archivo PDF."""
    reader = PdfReader(pdf_path)
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"--- Página {i + 1} ---\n{page_text}")
    return "\n\n".join(text_parts)


# ── Definición de secciones a procesar ─────────────────────────────────────────

SECTIONS = [
    {
        "id": "introduccion",
        "titulo": "Introducción y Motivación",
        "instrucciones": (
            "Redacta una introducción que explique:\n"
            "- El contexto general: qué son las GNNs y por qué son relevantes\n"
            "- El problema específico: las limitaciones de las GNNs estándar para "
            "grafos dirigidos y multigrafos (redes de transacciones financieras)\n"
            "- La motivación del trabajo: detección de delitos financieros como "
            "blanqueo de capitales\n"
            "- Un adelanto de las contribuciones principales\n\n"
            "NO traduzcas literalmente. Redacta con tus propias palabras."
        ),
    },
    {
        "id": "estado_del_arte",
        "titulo": "Estado del Arte",
        "instrucciones": (
            "Redacta una revisión del estado del arte que cubra:\n"
            "- El test de Weisfeiler-Lehman y las limitaciones de las MPNNs\n"
            "- Enfoques para superar estas limitaciones (k-WL, subgraph GNNs, "
            "features precalculadas, etc.)\n"
            "- Trabajos previos en GNNs para grafos dirigidos\n"
            "- Uso de GNNs en detección de fraude financiero\n\n"
            "Organiza el texto por temáticas, no sigas el mismo orden del paper. "
            "Usa tus propias palabras y estructura."
        ),
    },
    {
        "id": "metodologia",
        "titulo": "Metodología Propuesta",
        "instrucciones": (
            "Explica la metodología propuesta en el paper:\n"
            "- El concepto de Message Passing Neural Networks (MPNNs) y cómo "
            "funcionan\n"
            "- Las tres adaptaciones clave: reverse message passing, port numbering "
            "y ego IDs\n"
            "- Cómo se adaptan estas técnicas a multigrafos dirigidos\n"
            "- Los fundamentos teóricos: por qué la combinación permite detectar "
            "cualquier patrón de subgrafo dirigido\n\n"
            "Explica los conceptos de forma clara, como si el lector fuera un "
            "estudiante de máster. No copies fórmulas literalmente; explícalas "
            "conceptualmente."
        ),
    },
    {
        "id": "experimentos",
        "titulo": "Experimentos y Resultados",
        "instrucciones": (
            "Describe los experimentos y resultados:\n"
            "- Tareas sintéticas de detección de patrones de subgrafos\n"
            "- Dataset de blanqueo de capitales (IBM AML)\n"
            "- Dataset de phishing en Ethereum\n"
            "- Métricas utilizadas y resultados principales\n"
            "- Comparación con baselines (XGBoost, GNNs estándar, etc.)\n\n"
            "Destaca los resultados más relevantes con datos concretos pero "
            "reformulando la presentación. No copies tablas; describe los "
            "hallazgos narrativamente."
        ),
    },
    {
        "id": "conclusiones",
        "titulo": "Conclusiones y Trabajo Futuro",
        "instrucciones": (
            "Redacta unas conclusiones que incluyan:\n"
            "- Resumen de las contribuciones principales\n"
            "- Impacto práctico de los resultados (especialmente en detección de "
            "delitos financieros)\n"
            "- Limitaciones del enfoque\n"
            "- Posibles líneas de trabajo futuro\n"
            "- Relevancia para tu TFM sobre blanqueo de capitales con GNNs\n\n"
            "Añade una reflexión personal sobre cómo este trabajo se conecta "
            "con la investigación en el área."
        ),
    },
]


# ── Agente de redacción ───────────────────────────────────────────────────────

class ThesisWriterAgent:
    """Agente que procesa un paper científico y genera redacción académica original."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: La variable de entorno ANTHROPIC_API_KEY no está definida.")
            print("Ejecuta: export ANTHROPIC_API_KEY='tu-api-key'")
            sys.exit(1)

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def process_section(self, paper_text: str, section: dict) -> str:
        """Genera la redacción de una sección específica del TFM."""
        print(f"  Procesando: {section['titulo']}...")

        user_prompt = (
            f"A continuación tienes el texto completo de un paper científico. "
            f"Tu tarea es redactar la sección **\"{section['titulo']}\"** para un "
            f"Trabajo Fin de Máster en español.\n\n"
            f"### Instrucciones específicas para esta sección:\n"
            f"{section['instrucciones']}\n\n"
            f"### Texto del paper original:\n"
            f"{paper_text}\n\n"
            f"### Redacta ahora la sección \"{section['titulo']}\" en español, "
            f"con tus propias palabras:"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return response.content[0].text

    def generate_thesis_content(self, pdf_path: str) -> str:
        """Procesa el paper completo y genera todas las secciones del TFM."""
        print(f"\n{'='*70}")
        print(f"  AGENTE DE REDACCIÓN ACADÉMICA PARA TFM")
        print(f"{'='*70}")
        print(f"\n  PDF: {pdf_path}")
        print(f"  Modelo: {self.model}\n")

        # 1. Extraer texto del PDF
        print("  [1/3] Extrayendo texto del PDF...")
        paper_text = extract_text_from_pdf(pdf_path)
        print(f"         Extraídos {len(paper_text)} caracteres.\n")

        # 2. Procesar cada sección
        print("  [2/3] Generando redacción por secciones...\n")
        sections_output = []

        for section in SECTIONS:
            content = self.process_section(paper_text, section)
            sections_output.append({
                "titulo": section["titulo"],
                "contenido": content,
            })
            print(f"         ✓ {section['titulo']} completada.\n")

        # 3. Ensamblar documento final
        print("  [3/3] Ensamblando documento final...\n")
        final_doc = self._assemble_document(sections_output, pdf_path)

        return final_doc

    def _assemble_document(self, sections: list, pdf_path: str) -> str:
        """Ensambla todas las secciones en un documento markdown final."""
        lines = [
            "# Redes Neuronales de Grafos para Multigrafos Dirigidos: "
            "Aplicación a la Detección de Delitos Financieros",
            "",
            "> Redacción generada a partir del paper: *\"Provably Powerful Graph "
            "Neural Networks for Directed Multigraphs\"* (Egressy et al., AAAI 2024)",
            "",
            "> **Nota**: Este texto ha sido redactado con palabras propias a partir "
            "del paper original, evitando la traducción literal y el plagio. "
            "Las ideas y resultados se atribuyen a los autores originales.",
            "",
            "---",
            "",
        ]

        for section in sections:
            lines.append(f"## {section['titulo']}")
            lines.append("")
            lines.append(section["contenido"])
            lines.append("")
            lines.append("---")
            lines.append("")

        lines.extend([
            "## Referencias",
            "",
            "- Egressy, B., von Niederhäusern, L., Blanuša, J., Altman, E., "
            "Wattenhofer, R., & Atasu, K. (2024). Provably Powerful Graph Neural "
            "Networks for Directed Multigraphs. *Proceedings of the AAAI Conference "
            "on Artificial Intelligence*.",
            "",
        ])

        return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agente LLM para redactar contenido de TFM a partir de papers"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="gnn_tfm.pdf",
        help="Ruta al archivo PDF del paper (default: gnn_tfm.pdf)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="redaccion_tfm.md",
        help="Archivo de salida en markdown (default: redaccion_tfm.md)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Modelo de Claude a usar (default: claude-sonnet-4-20250514)",
    )
    args = parser.parse_args()

    # Verificar que el PDF existe
    if not Path(args.pdf).exists():
        print(f"Error: No se encontró el archivo '{args.pdf}'")
        sys.exit(1)

    # Crear y ejecutar el agente
    agent = ThesisWriterAgent(model=args.model)
    result = agent.generate_thesis_content(args.pdf)

    # Guardar resultado
    output_path = Path(args.output)
    output_path.write_text(result, encoding="utf-8")

    print(f"{'='*70}")
    print(f"  Documento generado: {output_path.absolute()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

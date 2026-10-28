"""Reuse existing atom segmentation and local-only embeddings, without forks.

Runtime must include the existing SkillEvolBench package on PYTHONPATH.
Imports are lazy so pure policy tests do not load model/runtime dependencies.
"""
import re

from .policy import rank_matches


def content_atoms(text):
    from skillevolbench.components.online_regularization import rule_segment

    # Frontmatter is retrieval metadata, not behavioral content.
    body = re.sub(r"\A---\r?\n.*?\r?\n---(?:\r?\n|\Z)", "", text,
                  count=1, flags=re.S)
    return [a.span for a in rule_segment(body)
            if a.span.strip() and not re.match(r"^\s*#{1,6}\s", a.span)]


def retrieve_old_skills(new_text, old_texts, *, tau, rho, top_k=3, audit=None):
    from skillevolbench.components.semantic_retrieval import cosine_similarity_matrix

    source = content_atoms(new_text)
    matrices = {
        path: cosine_similarity_matrix(source, content_atoms(text)).tolist()
        for path, text in sorted(old_texts.items())
    }
    return rank_matches(matrices, atom_count=len(source), tau=tau, rho=rho,
                        top_k=top_k, audit=audit)

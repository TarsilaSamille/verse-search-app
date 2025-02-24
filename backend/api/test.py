import json
import re
import math
from collections import defaultdict
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

# =============================================
# 1. Pré-processamento Aprimorado
# =============================================

def preprocess(sentence, is_target=False):
    """Lida com caracteres especiais em ambos os idiomas"""
    if is_target:
        # Preserva apóstrofo (ꞌ) e caracteres da língua alvo
        cleaned = re.sub(r'[^\w\sꞌ]', '', sentence.lower())
    else:
        # Pré-processamento padrão para inglês
        cleaned = re.sub(r"(\w+)'(\w+)", r'\1\2', sentence.lower())  # Contração
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
    return cleaned.split()

# =============================================
# 2. Modelo de Tradução Melhorado
# =============================================

def improved_translation_model(source_corpus, target_corpus):
    """Constrói modelo de tradução bidirecional com suavização"""
    translation_counts = defaultdict(lambda: defaultdict(float))

    # Contagem de ocorrências
    for src_sent, tgt_sent in zip(source_corpus, target_corpus):
        src_words = preprocess(src_sent)
        tgt_words = preprocess(tgt_sent, is_target=True)

        # Alinhamento direto
        for s in src_words:
            for t in tgt_words:
                translation_counts[s][t] += 1.0

        # Alinhamento reverso
        for t in tgt_words:
            for s in src_words:
                translation_counts[t][s] += 0.5  # Reforço bidirecional

    # Normalização para probabilidades
    translation_probs = {}
    for word, counts in translation_counts.items():
        total = sum(counts.values()) + 1e-10  # Suavização aditiva
        translation_probs[word] = {k: (v + 0.1)/total for k, v in counts.items()}  # Suavização de Laplace

    return translation_probs

# =============================================
# 3. Modelo de Linguagem Avançado
# =============================================

def train_advanced_lm(target_corpus):
    """Treina modelo de linguagem 4-gram com Kneser-Ney"""
    train_sents = [['<s>'] + preprocess(sent, is_target=True) + ['</s>'] for sent in target_corpus]
    train_data, vocab = padded_everygram_pipeline(4, train_sents)

    lm = KneserNeyInterpolated(4)
    lm.fit(train_data, vocab)
    return lm

# =============================================
# 4. Tradução Sensível ao Contexto
# =============================================

def context_sensitive_translate(sentence, trans_model, lm, beam_width=5, alpha=0.7):
    """Tradução com busca em feixe e decodificação contextual"""
    source_words = preprocess(sentence)
    if not source_words:
        return ""

    # Inicialização dos feixes
    beams = [(['<s>'], '<s>', 0.0)]

    for src_word in source_words:
        candidates = []
        possible_trans = trans_model.get(src_word, {'[UNK]': 1e-10})

        # Expansão dos feixes
        for seq, prev_word, score in beams:
            for trans_word, trans_prob in possible_trans.items():
                # Probabilidade do modelo de linguagem
                context = seq[-3:]  # Contexto de 3 palavras
                lm_prob = max(lm.score(trans_word, context[-2:]), 1e-10)

                # Cálculo seguro do score
                safe_trans = max(trans_prob, 1e-10)
                new_score = score + (
                    math.log(safe_trans) +
                    alpha * math.log(lm_prob)
                
                new_seq = seq + [trans_word]
                candidates.append((new_seq, trans_word, new_score))

        # Seleção dos melhores feixes
        candidates.sort(key=lambda x: -x[2])
        beams = candidates[:beam_width]

    # Processamento do resultado final
    best_sequence = max(beams, key=lambda x: x[2])[0][1:]
    return ' '.join([w for w in best_sequence if w not in ['</s>', '<s>']])

# =============================================
# 5. Execução Principal
# =============================================

if __name__ == "__main__":
    # Carrega dados
    with open('jagoy-english.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepara corpora paralelos
    source_corpus = [item["translation"]["en"] for item in data]
    target_corpus = [item["translation"]["bj"] for item in data]

    # Treina modelos
    print("Treinando modelo de tradução...")
    trans_model = improved_translation_model(source_corpus, target_corpus)

    print("Treinando modelo de linguagem...")
    lm = train_advanced_lm(target_corpus)

    # Teste
    test_sentence = "Jesus said don't worry about what you will eat."
    translation = context_sensitive_translate(test_sentence, trans_model, lm)

    print("\nEntrada:", test_sentence)
    print("Tradução:", translation)
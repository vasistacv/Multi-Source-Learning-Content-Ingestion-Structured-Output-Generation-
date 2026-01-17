from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
import re
from loguru import logger

from ..nlp.nlp_engine import NLPEngine


@dataclass
class Flashcard:
    question: str
    answer: str
    topic: str
    difficulty: str
    tags: List[str]
    confidence_score: float
    source: str


@dataclass
class QuizQuestion:
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    topic: str
    difficulty: str
    question_type: str


class FlashcardGenerator:
    def __init__(self, nlp_engine: NLPEngine):
        self.nlp_engine = nlp_engine
    
    def generate(
        self,
        text: str,
        num_cards: int = 20,
        min_confidence: float = 0.7
    ) -> List[Flashcard]:
        logger.info(f"Generating {num_cards} flashcards")
        
        flashcards = []
        
        qa_pairs = self._extract_definition_pairs(text)
        flashcards.extend(qa_pairs)
        
        concept_cards = self._generate_concept_cards(text)
        flashcards.extend(concept_cards)
        
        entity_cards = self._generate_entity_cards(text)
        flashcards.extend(entity_cards)
        
        flashcards = [fc for fc in flashcards if fc.confidence_score >= min_confidence]
        
        flashcards.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return flashcards[:num_cards]
    
    def _extract_definition_pairs(self, text: str) -> List[Flashcard]:
        flashcards = []
        
        definition_patterns = [
            r'(.+?)\s+is\s+(?:defined as|known as)\s+(.+?)\.',
            r'(.+?)\s+refers to\s+(.+?)\.',
            r'(.+?)\s+means\s+(.+?)\.',
            r'(?:The term|The concept of)\s+(.+?)\s+(?:is|means)\s+(.+?)\.',
        ]
        
        doc = self.nlp_engine.nlp(text[:500000])
        
        for sent in doc.sents:
            sent_text = sent.text
            
            for pattern in definition_patterns:
                matches = re.finditer(pattern, sent_text, re.IGNORECASE)
                for match in matches:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    
                    if 5 <= len(term.split()) <= 10 and len(definition.split()) >= 5:
                        flashcard = Flashcard(
                            question=f"What is {term}?",
                            answer=definition,
                            topic=self._extract_topic(term),
                            difficulty=self._estimate_difficulty(term, definition),
                            tags=[term],
                            confidence_score=0.85,
                            source="definition_extraction"
                        )
                        flashcards.append(flashcard)
        
        return flashcards
    
    def _generate_concept_cards(self, text: str) -> List[Flashcard]:
        flashcards = []
        
        concepts = self.nlp_engine.extract_key_concepts(text, num_concepts=30)
        
        doc = self.nlp_engine.nlp(text[:500000])
        sentences = list(doc.sents)
        
        for concept_dict in concepts[:15]:
            concept = concept_dict['concept']
            score = concept_dict['score']
            
            relevant_sentences = [
                sent.text for sent in sentences
                if concept.lower() in sent.text.lower()
            ][:3]
            
            if relevant_sentences:
                context = " ".join(relevant_sentences)
                
                flashcard = Flashcard(
                    question=f"Explain the concept of '{concept}'",
                    answer=context[:300],
                    topic=concept,
                    difficulty=self._estimate_difficulty_from_score(score),
                    tags=[concept],
                    confidence_score=min(score * 1.5, 0.95),
                    source="concept_extraction"
                )
                flashcards.append(flashcard)
        
        return flashcards
    
    def _generate_entity_cards(self, text: str) -> List[Flashcard]:
        flashcards = []
        
        entities = self.nlp_engine.extract_entities(text)
        
        entity_dict = {}
        for ent in entities:
            key = (ent['text'], ent['label'])
            if key not in entity_dict:
                entity_dict[key] = []
            entity_dict[key].append(ent)
        
        doc = self.nlp_engine.nlp(text[:500000])
        
        for (ent_text, ent_label), ent_list in entity_dict.items():
            if len(ent_list) >= 2:
                relevant_sentences = [
                    sent.text for sent in doc.sents
                    if ent_text in sent.text
                ][:2]
                
                if relevant_sentences:
                    context = " ".join(relevant_sentences)
                    
                    flashcard = Flashcard(
                        question=f"What is '{ent_text}' ({ent_label})?",
                        answer=context[:300],
                        topic=ent_label,
                        difficulty="medium",
                        tags=[ent_text, ent_label],
                        confidence_score=0.75,
                        source="entity_extraction"
                    )
                    flashcards.append(flashcard)
        
        return flashcards
    
    def _extract_topic(self, text: str) -> str:
        words = text.split()
        return words[0].capitalize() if words else "General"
    
    def _estimate_difficulty(self, term: str, definition: str) -> str:
        complexity = len(term.split()) + len(definition.split()) / 10
        
        if complexity < 10:
            return "easy"
        elif complexity < 20:
            return "medium"
        else:
            return "hard"
    
    def _estimate_difficulty_from_score(self, score: float) -> str:
        if score > 0.7:
            return "easy"
        elif score > 0.4:
            return "medium"
        else:
            return "hard"


class QuizGenerator:
    def __init__(self, nlp_engine: NLPEngine):
        self.nlp_engine = nlp_engine
    
    def generate(
        self,
        text: str,
        num_questions: int = 10
    ) -> List[QuizQuestion]:
        logger.info(f"Generating {num_questions} quiz questions")
        
        questions = []
        
        mcq = self._generate_mcq(text, num_questions // 2)
        questions.extend(mcq)
        
        tf_questions = self._generate_true_false(text, num_questions // 4)
        questions.extend(tf_questions)
        
        fill_blanks = self._generate_fill_in_blanks(text, num_questions // 4)
        questions.extend(fill_blanks)
        
        random.shuffle(questions)
        return questions[:num_questions]
    
    def _generate_mcq(self, text: str, num_questions: int) -> List[QuizQuestion]:
        questions = []
        
        concepts = self.nlp_engine.extract_key_concepts(text, num_concepts=num_questions * 2)
        entities = self.nlp_engine.extract_entities(text)
        
        doc = self.nlp_engine.nlp(text[:500000])
        
        for concept_dict in concepts[:num_questions]:
            concept = concept_dict['concept']
            
            relevant_sentences = [
                sent.text for sent in doc.sents
                if concept.lower() in sent.text.lower()
            ]
            
            if relevant_sentences:
                context = relevant_sentences[0]
                
                question_text = f"Which of the following best describes '{concept}'?"
                
                correct_answer = context[:150]
                
                distractors = self._generate_distractors(concept, entities, concepts, num=3)
                
                options = [correct_answer] + distractors
                random.shuffle(options)
                
                quiz_question = QuizQuestion(
                    question=question_text,
                    options=options,
                    correct_answer=correct_answer,
                    explanation=context,
                    topic=concept,
                    difficulty="medium",
                    question_type="multiple_choice"
                )
                questions.append(quiz_question)
        
        return questions
    
    def _generate_true_false(self, text: str, num_questions: int) -> List[QuizQuestion]:
        questions = []
        
        doc = self.nlp_engine.nlp(text[:500000])
        sentences = list(doc.sents)
        
        for sent in sentences[:num_questions * 2]:
            if len(sent.text.split()) > 10:
                is_true = random.choice([True, False])
                
                if is_true:
                    statement = sent.text
                    correct = "True"
                    explanation = "This statement is directly from the source material."
                else:
                    statement = self._negate_statement(sent.text)
                    correct = "False"
                    explanation = f"The correct statement is: {sent.text}"
                
                quiz_question = QuizQuestion(
                    question=statement,
                    options=["True", "False"],
                    correct_answer=correct,
                    explanation=explanation,
                    topic="General",
                    difficulty="easy",
                    question_type="true_false"
                )
                questions.append(quiz_question)
                
                if len(questions) >= num_questions:
                    break
        
        return questions
    
    def _generate_fill_in_blanks(self, text: str, num_questions: int) -> List[QuizQuestion]:
        questions = []
        
        entities = self.nlp_engine.extract_entities(text)
        doc = self.nlp_engine.nlp(text[:500000])
        
        for ent in entities[:num_questions * 2]:
            ent_text = ent['text']
            
            relevant_sentences = [
                sent.text for sent in doc.sents
                if ent_text in sent.text and len(sent.text.split()) > 8
            ]
            
            if relevant_sentences:
                sentence = relevant_sentences[0]
                question_text = sentence.replace(ent_text, "_____")
                
                quiz_question = QuizQuestion(
                    question=f"Fill in the blank: {question_text}",
                    options=[ent_text],
                    correct_answer=ent_text,
                    explanation=f"The complete sentence is: {sentence}",
                    topic=ent['label'],
                    difficulty="medium",
                    question_type="fill_in_blank"
                )
                questions.append(quiz_question)
                
                if len(questions) >= num_questions:
                    break
        
        return questions
    
    def _generate_distractors(
        self,
        concept: str,
        entities: List[Dict],
        concepts: List[Dict],
        num: int = 3
    ) -> List[str]:
        distractors = []
        
        same_type_entities = [
            ent['text'] for ent in entities
            if ent['text'].lower() != concept.lower()
        ]
        
        other_concepts = [
            c['concept'] for c in concepts
            if c['concept'].lower() != concept.lower()
        ]
        
        candidates = same_type_entities + other_concepts
        
        if len(candidates) >= num:
            distractors = random.sample(candidates, num)
        else:
            distractors = candidates + [f"Not {concept}"] * (num - len(candidates))
        
        return distractors
    
    def _negate_statement(self, statement: str) -> str:
        negations = {
            ' is ': ' is not ',
            ' are ': ' are not ',
            ' was ': ' was not ',
            ' were ': ' were not ',
            ' can ': ' cannot ',
            ' will ': ' will not ',
            ' should ': ' should not ',
        }
        
        for pos, neg in negations.items():
            if pos in statement.lower():
                return statement.replace(pos, neg, 1)
        
        return f"It is incorrect that {statement.lower()}"

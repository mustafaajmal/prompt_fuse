#!/usr/bin/env python3
"""Generate synthetic paraphrase dataset for unifier evaluation.

Design principles:
    - Each cluster contains prompts that genuinely rephrase the same intent
      using different sentence structures, vocabulary, and framing — not
      just prefix/suffix wrappers around the same core string.
    - Expansion templates preserve structural diversity: they vary clause
      order, voice (active/passive), specificity, and register.
    - Near-duplicate detection removes variants with high token overlap so
      the unifier can't cheat with surface-level matching.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed clusters: each list contains manually-crafted structural paraphrases
# that a human would recognise as the same instruction but that differ at
# the token level far more than "Please {X}" vs "Task: {X}".
# ---------------------------------------------------------------------------

PARAPHRASE_CLUSTERS: list[tuple[str, list[str]]] = [
    # ---- summarization (6 clusters) ----------------------------------------
    ("summarization_3sent", [
        "Summarize the following paragraph in three sentences.",
        "Read the passage and distill it down to exactly three sentences.",
        "I need the key points from this paragraph — give me a three-sentence version.",
        "Take the text below and produce a concise three-sentence recap.",
        "Boil this paragraph down to its essentials in no more than three sentences.",
        "After reading the following, write three sentences that capture the main ideas.",
        "Your job is to compress the paragraph into a three-sentence summary.",
        "What are the three most important sentences you could write to represent this paragraph?",
        "Reduce the following to three sentences without losing the core meaning.",
        "Draft a three-sentence abstract of the passage provided.",
    ]),
    ("summarization_1para", [
        "Summarize this article in one paragraph.",
        "Write a single cohesive paragraph that captures the article's main argument.",
        "Condense the article into a paragraph-length overview.",
        "Give me the gist of this article in one short paragraph.",
        "I'd like a one-paragraph version of the article below.",
        "Produce a brief paragraph summarizing the key takeaways from this article.",
        "Read the article and respond with a single-paragraph summary.",
        "Can you shrink this article into just one paragraph?",
        "In your own words, summarize the entire article in a single paragraph.",
        "Provide a paragraph that a busy reader could use instead of reading the full article.",
    ]),
    ("summarization_bullet", [
        "List the main points of this document as bullet points.",
        "Extract the key ideas and present them as a bulleted list.",
        "Turn this document into a concise set of bullet points.",
        "What are the most important points? Use bullet format.",
        "Break the document down into its core arguments, one bullet each.",
        "Give me a bullet-point outline of the document's content.",
        "Read through this and pull out the highlights as bullets.",
        "Produce a bulleted summary covering every major point in the text.",
        "Distill this into bullets — one per key idea.",
        "Summarize using short, punchy bullet points.",
    ]),
    ("summarization_eli5", [
        "Explain the main idea of this text as if I were five years old.",
        "Simplify the passage below so a young child could understand it.",
        "Can you dumb this down to a kindergarten-level explanation?",
        "Restate the core message in the simplest language possible.",
        "Pretend I know nothing about this topic — explain it simply.",
        "Give me the ELI5 version of this text.",
        "How would you explain this passage to someone with no background knowledge?",
        "Strip away the jargon and tell me what this means in plain words.",
        "Break this down into the simplest possible terms.",
        "Translate this into everyday language a child would follow.",
    ]),
    ("summarization_tldr", [
        "Give me a one-sentence TL;DR of this.",
        "What's the single most important takeaway from this text?",
        "If you had to capture this in one sentence, what would it be?",
        "Reduce everything below to a single sentence.",
        "One-line summary, please.",
        "Compress the entire passage into a single declarative sentence.",
        "Summarize this in as few words as possible — one sentence max.",
        "What's the bottom line here? One sentence.",
        "Distill this to its absolute essence in one sentence.",
        "Express the main point in exactly one sentence.",
    ]),
    ("summarization_academic", [
        "Write an academic abstract for the research described below.",
        "Draft a structured abstract covering background, methods, results, and conclusion.",
        "Produce a formal abstract suitable for a journal submission.",
        "Summarize this research in abstract format following academic conventions.",
        "Create a 150-word abstract that a reviewer would find acceptable.",
        "Compose a scholarly abstract for the study described in the passage.",
        "Write a publication-ready abstract for the following research.",
        "Generate an abstract with objective, methodology, findings, and implications.",
        "Prepare a concise abstract that adheres to academic writing standards.",
        "Formulate an abstract capturing the study's purpose, approach, and key results.",
    ]),

    # ---- classification (5 clusters) ---------------------------------------
    ("classification_sentiment", [
        "Classify the sentiment of the following review as positive, negative, or neutral.",
        "Read this review and tell me whether the overall tone is positive, negative, or neutral.",
        "What sentiment does the reviewer express? Choose from: positive, negative, neutral.",
        "Determine the emotional valence of the review — is it favorable, unfavorable, or mixed?",
        "Label this review's sentiment: positive, negative, or neutral.",
        "Would you say the reviewer's opinion is positive, negative, or somewhere in between?",
        "Assess whether this review leans positive, negative, or stays neutral.",
        "Judge the sentiment expressed in the following customer review.",
        "Is the tone of this review appreciative, critical, or indifferent?",
        "Categorize the following feedback as positive, negative, or neutral sentiment.",
    ]),
    ("classification_spam", [
        "Is this email spam or not spam?",
        "Classify the email below as spam or legitimate.",
        "Determine whether the following message is unwanted bulk mail.",
        "Would this email belong in the spam folder or the inbox?",
        "Label this email: spam or ham.",
        "Does this message look like spam to you? Answer spam or not spam.",
        "Evaluate whether the following email is a genuine message or junk.",
        "Should this email be filtered as spam? Yes or no.",
        "Classify: is the email below solicited correspondence or spam?",
        "Read the email and decide if it's spam or a real message.",
    ]),
    ("classification_topic", [
        "Assign this article to one of the following categories: politics, sports, technology, entertainment, science.",
        "Which topic best describes this article? Options: politics, sports, tech, entertainment, science.",
        "Categorize the article into the most fitting topic area.",
        "Determine the primary subject matter of this article from the given categories.",
        "What section of a newspaper would this article appear in?",
        "Label the article's topic: politics, sports, technology, entertainment, or science.",
        "Read the article and select the category that best matches its content.",
        "Identify the dominant theme of this article from the options provided.",
        "If you were an editor, which desk would handle this article?",
        "Classify the following article by its primary subject area.",
    ]),
    ("classification_intent", [
        "Identify the user's intent from the message below.",
        "What is the customer trying to accomplish in this message?",
        "Determine the primary intent behind this user query.",
        "Classify the purpose of the following message.",
        "What action is the user requesting in this message?",
        "Read the customer message and categorize their intent.",
        "What does the user want? Identify the underlying intent.",
        "Parse the following message for the user's main goal.",
        "Determine what the sender is asking for in this message.",
        "Classify the intent: is the user asking, complaining, requesting, or informing?",
    ]),
    ("classification_toxicity", [
        "Is the following comment toxic or non-toxic?",
        "Rate whether this comment contains harmful or abusive language.",
        "Determine if the comment below would violate community guidelines.",
        "Label this comment as toxic or acceptable.",
        "Does this comment contain hate speech, threats, or abusive content?",
        "Assess the toxicity level of the following user comment.",
        "Would a content moderator flag this comment? Answer toxic or non-toxic.",
        "Evaluate whether the language in this comment is harmful.",
        "Classify: is this comment civil discourse or toxic behavior?",
        "Judge if this comment crosses the line into toxicity.",
    ]),

    # ---- translation (4 clusters) ------------------------------------------
    ("translation_en_fr", [
        "Translate the following text from English to French.",
        "Convert the English passage below into French.",
        "Render this text in French.",
        "Provide a French translation of the following English text.",
        "How would you say this in French?",
        "Rewrite the passage below in French.",
        "Translate to French, preserving the original meaning and tone.",
        "I need this in French — please translate.",
        "Produce a faithful French version of the text.",
        "Turn the English below into natural-sounding French.",
    ]),
    ("translation_en_es", [
        "Translate the following sentence to Spanish.",
        "Render the sentence below in Spanish.",
        "Provide a Spanish translation of the following.",
        "How would a native Spanish speaker say this?",
        "Convert this English sentence into Spanish.",
        "Rewrite in Spanish, keeping the meaning intact.",
        "I need a Spanish version of this sentence.",
        "Translate to Spanish with natural phrasing.",
        "Express the same idea in Spanish.",
        "Put this into Spanish for me.",
    ]),
    ("translation_en_de", [
        "Translate the following into German.",
        "Give me the German equivalent of this text.",
        "Render this passage in German.",
        "Convert to German while preserving nuance.",
        "Provide a German translation.",
        "How would this read in German?",
        "Rewrite in German, maintaining the original register.",
        "I need this translated to German.",
        "Produce a natural German version of the text below.",
        "Express the following in German.",
    ]),
    ("translation_en_zh", [
        "Translate the following text into Mandarin Chinese.",
        "Convert this passage to simplified Chinese.",
        "Render the text below in Chinese.",
        "Provide a Chinese translation of the following.",
        "How would this be expressed in Mandarin?",
        "Rewrite in Chinese, keeping the tone formal.",
        "Translate to simplified Chinese characters.",
        "I need a Mandarin version of this.",
        "Put this into natural-sounding Chinese.",
        "Produce a faithful Chinese translation.",
    ]),

    # ---- question answering (3 clusters) -----------------------------------
    ("qa_extractive", [
        "Answer the question based on the context provided.",
        "Using the context below, find the answer to the question.",
        "Read the passage and respond to the question with information from the text.",
        "The answer is somewhere in the context — extract it.",
        "Based solely on the provided context, answer the following question.",
        "Find the relevant information in the passage to answer this question.",
        "What does the context say about the question asked below?",
        "Locate the answer within the text and state it clearly.",
        "Answer using only information present in the given context.",
        "Refer to the passage to answer the following.",
    ]),
    ("qa_open", [
        "Answer the following question to the best of your knowledge.",
        "What do you know about the question below? Give a thorough answer.",
        "Provide a detailed response to this question.",
        "Use your knowledge to answer the following.",
        "Give me a comprehensive answer to this question.",
        "Respond to the question below with as much detail as you can.",
        "What is the answer to the following? Explain your reasoning.",
        "Answer this question thoroughly and accurately.",
        "Please address the following question in detail.",
        "Share everything you know that's relevant to this question.",
    ]),
    ("qa_boolean", [
        "Answer yes or no: does the passage support the claim below?",
        "Based on the text, is the following statement true or false?",
        "Does the context confirm or deny the claim? Answer true/false.",
        "Read the passage and determine if the statement is supported.",
        "Is the following claim consistent with the information provided? Yes or no.",
        "Verify: does the text support this assertion?",
        "True or false — is the statement below backed by the passage?",
        "Check whether the passage provides evidence for the claim.",
        "Answer with yes or no: is the claim supported by the context?",
        "Determine the veracity of the statement given the passage.",
    ]),

    # ---- code generation (3 clusters) --------------------------------------
    ("code_python_func", [
        "Write a Python function that implements the following specification.",
        "Generate Python code for the task described below.",
        "Create a Python function matching this specification.",
        "Implement the described behavior as a Python function.",
        "Code this up in Python — write a function that does what's described.",
        "I need a Python function that fulfills the following requirements.",
        "Write the Python implementation for the spec below.",
        "Draft a Python function according to these instructions.",
        "Produce working Python code that satisfies the specification.",
        "Build a Python function based on the description provided.",
    ]),
    ("code_debug", [
        "Find and fix the bug in the following code.",
        "This code has an error — identify it and provide the corrected version.",
        "Debug the code below and explain what was wrong.",
        "What's wrong with this code? Fix it.",
        "Identify the bug and supply a working version.",
        "The following code doesn't work as expected — diagnose and repair it.",
        "Locate the defect in this code and provide a fix.",
        "Review the code for errors and return a corrected version.",
        "Something is broken in this code — find it and fix it.",
        "Debug this snippet and explain the root cause of the failure.",
    ]),
    ("code_explain", [
        "Explain what the following code does, line by line.",
        "Walk me through this code and explain each part.",
        "Provide a detailed explanation of how this code works.",
        "What does this code do? Break it down step by step.",
        "Describe the purpose and logic of each section of this code.",
        "Annotate the code below with explanations of what each line does.",
        "I don't understand this code — can you explain it to me?",
        "Give a thorough walkthrough of the code's behavior.",
        "Explain the control flow and logic in this code snippet.",
        "Help me understand this code by explaining it in plain English.",
    ]),

    # ---- extraction (3 clusters) -------------------------------------------
    ("extraction_ner", [
        "Extract all named entities from the following text.",
        "Identify every person, organization, and location mentioned in the passage.",
        "List the named entities found in this text.",
        "Pull out all proper nouns and named entities from the passage.",
        "What people, places, and organizations are mentioned here?",
        "Find and list every named entity in the following.",
        "Scan the text for named entities and enumerate them.",
        "Identify all entities of type person, org, and location in this text.",
        "Extract proper names, company names, and place names from the passage.",
        "Parse the text below for named entities and return them as a list.",
    ]),
    ("extraction_key_value", [
        "Extract the key-value pairs from the following text.",
        "Parse the text and return structured field-value pairs.",
        "Identify all fields and their corresponding values in this passage.",
        "Turn the unstructured text below into a set of key-value pairs.",
        "What structured data can you extract from this text?",
        "Convert the information in this passage to key-value format.",
        "Pull out the important fields and their values from the text.",
        "Read the text and organize the data into field: value pairs.",
        "Extract structured information from the following unstructured text.",
        "Map the relevant data points in this passage to labeled fields.",
    ]),
    ("extraction_table", [
        "Extract the data from this text and format it as a table.",
        "Turn the information below into a structured table.",
        "Parse the passage and present the data in tabular form.",
        "Organize the facts in this text into rows and columns.",
        "Create a table from the data described in the following text.",
        "Can you put this information into a table format?",
        "Structure the data from the passage as a markdown table.",
        "Represent the key data points below in table form.",
        "Read the text and output the information as a formatted table.",
        "Convert the narrative data into a clean table.",
    ]),

    # ---- rewriting (3 clusters) --------------------------------------------
    ("rewriting_formal", [
        "Rewrite the following sentence to be more formal.",
        "Make the sentence below more professional in tone.",
        "Rephrase this in formal, business-appropriate language.",
        "Elevate the register of this sentence to formal English.",
        "Convert this casual sentence into formal writing.",
        "How would this sentence sound in a formal report?",
        "Recast this sentence using more formal vocabulary and structure.",
        "Polish this sentence for a professional audience.",
        "Transform the informal phrasing into something suitable for official communication.",
        "Rewrite with a formal tone — no slang or casual language.",
    ]),
    ("rewriting_concise", [
        "Make this paragraph more concise without losing meaning.",
        "Tighten the writing — remove unnecessary words while keeping the point.",
        "Edit this for brevity. Keep the core message, cut the filler.",
        "Rewrite this paragraph to be half its current length.",
        "Shorten this without sacrificing clarity or meaning.",
        "Trim the fat from this paragraph.",
        "Condense the following while preserving all key information.",
        "Make this more succinct — every word should earn its place.",
        "Reduce wordiness in the paragraph below.",
        "Rewrite to be as brief as possible without changing the meaning.",
    ]),
    ("rewriting_tone_casual", [
        "Rewrite this in a casual, conversational tone.",
        "Make this sound like something you'd say to a friend.",
        "Convert this formal text into everyday, relaxed language.",
        "Loosen up the tone — make it sound natural and informal.",
        "How would you say this casually? Rewrite it that way.",
        "Take the stiffness out of this and make it conversational.",
        "Rephrase in a friendly, approachable tone.",
        "Rewrite as if you're texting a colleague you're close with.",
        "Drop the formality and rewrite this in plain, casual English.",
        "Make this read like a blog post, not a legal document.",
    ]),

    # ---- reasoning (3 clusters) --------------------------------------------
    ("reasoning_math", [
        "Solve the following math word problem step by step.",
        "Work through this math problem showing each step.",
        "Provide a step-by-step solution to the problem below.",
        "Walk me through the solution to this math problem.",
        "Show your work as you solve the following problem.",
        "Solve this problem and explain your reasoning at each step.",
        "Break this math problem into steps and solve each one.",
        "I need a detailed, step-by-step solution to this problem.",
        "Work out the answer to this problem, showing all intermediate steps.",
        "Explain how to solve this math problem from start to finish.",
    ]),
    ("reasoning_logic", [
        "Evaluate the logical validity of the following argument.",
        "Is this argument logically sound? Explain why or why not.",
        "Analyze the reasoning in the argument below.",
        "Does this argument contain any logical fallacies?",
        "Assess whether the conclusion follows from the premises.",
        "Check the logic of this argument and identify any flaws.",
        "Is the reasoning here valid? Point out any weaknesses.",
        "Critique the logical structure of the following argument.",
        "Determine if the conclusion is properly supported by the premises.",
        "Evaluate this argument for logical consistency and soundness.",
    ]),
    ("reasoning_causal", [
        "What are the likely causes of the situation described below?",
        "Identify the causal factors behind the scenario in the passage.",
        "Explain what probably led to the outcome described.",
        "Analyze the root causes of the situation described in the text.",
        "What factors most likely contributed to this outcome?",
        "Trace the causal chain that led to the situation described.",
        "Provide a causal analysis of the scenario below.",
        "Why did this happen? Identify the most plausible causes.",
        "Determine the underlying causes of the described event.",
        "Offer a causal explanation for the outcome described in the passage.",
    ]),

    # ---- brainstorming (2 clusters) ----------------------------------------
    ("brainstorming_ideas", [
        "Brainstorm ten ideas for the topic below.",
        "Generate 10 creative ideas related to this topic.",
        "Come up with ten different approaches to the following.",
        "List ten possible ideas for the following subject.",
        "Give me 10 distinct ideas inspired by this topic.",
        "Think of ten creative angles on the subject below.",
        "I need ten ideas — go wide and be creative.",
        "Produce ten varied ideas related to the following theme.",
        "Brainstorm broadly: give me 10 ideas for this.",
        "Generate a diverse set of ten ideas on this topic.",
    ]),
    ("brainstorming_names", [
        "Suggest ten names for the product described below.",
        "Come up with ten brand name ideas for this product.",
        "Brainstorm ten catchy names for the following product.",
        "Generate ten potential names for this product or service.",
        "What are ten good names for the product described?",
        "I need ten name candidates for this product — be creative.",
        "List ten possible product names based on the description.",
        "Think of ten memorable names for the product below.",
        "Propose ten naming options for this product.",
        "Dream up ten names that would work for this product.",
    ]),

    # ---- comparison (2 clusters) -------------------------------------------
    ("comparison_passages", [
        "Compare and contrast the two passages below.",
        "Discuss the similarities and differences between these two texts.",
        "How are these two passages alike, and how do they differ?",
        "Provide a comparison of the following two excerpts.",
        "Analyze the two passages for points of agreement and disagreement.",
        "What do these passages have in common, and where do they diverge?",
        "Draw a comparison between the two texts below.",
        "Examine both passages and highlight their similarities and differences.",
        "Contrast the arguments or perspectives in these two passages.",
        "Write a comparative analysis of the following two texts.",
    ]),
    ("comparison_options", [
        "Compare the following options and recommend the best one.",
        "Evaluate these alternatives and tell me which is strongest.",
        "Which of the options below is the best choice, and why?",
        "Weigh the pros and cons of each option and give your recommendation.",
        "Analyze these options side by side and pick a winner.",
        "Help me decide between these options — compare them thoroughly.",
        "Assess each option and recommend the one you'd choose.",
        "Rank the following options from best to worst with reasoning.",
        "Which option offers the best tradeoff? Explain your reasoning.",
        "Compare these choices and advise me on the optimal selection.",
    ]),

    # ---- data analysis (2 clusters) ----------------------------------------
    ("analysis_trends", [
        "Identify the key trends in the data below.",
        "What patterns do you see in this dataset?",
        "Analyze the data and highlight the most significant trends.",
        "Look at the numbers below and tell me what trends stand out.",
        "Summarize the main trends visible in this data.",
        "What story does this data tell? Identify the trends.",
        "Examine the data for notable patterns or trends.",
        "Describe the trajectory and patterns in the following data.",
        "Point out the most important trends in the dataset.",
        "Analyze this data and report on any emerging patterns.",
    ]),
    ("analysis_anomalies", [
        "Identify any anomalies or outliers in the data below.",
        "Are there any unusual data points in this dataset?",
        "Flag anything in this data that looks abnormal.",
        "Scan the data for outliers and explain why they stand out.",
        "Which values in this dataset are anomalous?",
        "Detect any irregularities in the following data.",
        "Point out data points that deviate significantly from the norm.",
        "Identify suspicious or unexpected values in the dataset.",
        "Are there outliers here? If so, what might explain them?",
        "Examine this data for statistical anomalies.",
    ]),

    # ---- instruction following (3 clusters) --------------------------------
    ("instruction_format_json", [
        "Return your response as a JSON object with the specified fields.",
        "Format your output as valid JSON matching the schema below.",
        "Respond with a JSON object — no prose, just the structured data.",
        "Give me the answer as a JSON document with the fields listed.",
        "Your response should be a well-formed JSON object.",
        "Output the result in JSON format using the keys described.",
        "Please structure your response as JSON.",
        "Return only a JSON object — no explanations or markdown.",
        "I need the output in JSON. Follow the schema provided.",
        "Produce a JSON response conforming to the format specified.",
    ]),
    ("instruction_format_markdown", [
        "Format your response using markdown with headers and bullet points.",
        "Use markdown formatting: headers, lists, bold for emphasis.",
        "Structure your answer with markdown headings and organized sections.",
        "Present the response in well-formatted markdown.",
        "Use proper markdown syntax with ## headers and - bullets.",
        "Write your response as a formatted markdown document.",
        "Organize your answer using markdown headers and lists.",
        "Apply markdown formatting to make the response scannable.",
        "Respond with clean markdown — headers, bullets, code blocks as needed.",
        "Structure this as a markdown document with clear sections.",
    ]),
    ("instruction_persona", [
        "Respond as if you are an experienced senior software engineer.",
        "Answer from the perspective of a senior developer with 15 years of experience.",
        "Take on the role of a seasoned software engineer when responding.",
        "Pretend you are a principal engineer reviewing this for a junior developer.",
        "Adopt the persona of an experienced technical lead.",
        "Answer this the way a senior staff engineer would at a top tech company.",
        "Respond as a veteran software engineer would — practical and direct.",
        "Channel an experienced developer — give me the kind of answer they'd give.",
        "Play the role of a senior engineer mentoring a junior colleague.",
        "Imagine you're a 15-year engineering veteran responding to this.",
    ]),

    # ---- editing (2 clusters) ----------------------------------------------
    ("editing_grammar", [
        "Fix the grammar and spelling errors in the following text.",
        "Proofread this text and correct any grammatical mistakes.",
        "Edit the passage for grammar, punctuation, and spelling.",
        "Clean up the writing — fix any errors in grammar or spelling.",
        "Correct all grammatical and typographical errors in the text.",
        "Review and fix the language errors in the following.",
        "Polish the grammar and fix any typos in this text.",
        "Copy-edit the passage below for correctness.",
        "Check the text for grammatical errors and return a corrected version.",
        "Identify and fix every grammar or spelling mistake in this passage.",
    ]),
    ("editing_clarity", [
        "Rewrite this passage to improve clarity and readability.",
        "Edit the text below so it's easier to understand.",
        "Make this passage clearer without changing the meaning.",
        "Improve the readability of the following text.",
        "Revise for clarity — the text should be easy to follow.",
        "Rework this passage so the ideas come through more clearly.",
        "Simplify the structure and wording to improve comprehension.",
        "The writing is confusing — rewrite it to be clear and direct.",
        "Edit for clarity: remove ambiguity and tighten the prose.",
        "Restructure this text so a general audience can follow it easily.",
    ]),

    # ---- creative writing (2 clusters) -------------------------------------
    ("creative_story", [
        "Write a short story based on the following prompt.",
        "Create a piece of short fiction inspired by the scenario below.",
        "Compose a brief narrative using this as your starting point.",
        "Use the prompt below as the basis for a short story.",
        "Write a creative short story from this prompt.",
        "Craft a fictional narrative based on the setup described.",
        "Develop a short story around the concept described below.",
        "Turn this prompt into a compelling piece of short fiction.",
        "Let the following inspire a short story — write it out.",
        "Produce an original short story based on the scenario provided.",
    ]),
    ("creative_poetry", [
        "Write a poem about the subject described below.",
        "Compose a poem inspired by the following theme.",
        "Create a piece of poetry on this topic.",
        "Write a poem that captures the essence of the subject below.",
        "Produce an original poem related to this theme.",
        "Craft a poem about the topic described.",
        "Express the following subject through poetry.",
        "Write a poem — any form — on the theme provided.",
        "Channel the subject below into a poem.",
        "Compose verses inspired by the following topic.",
    ]),

    # ---- safety / moderation (2 clusters) ----------------------------------
    ("moderation_content_policy", [
        "Does the following text violate any content policies? Explain.",
        "Review this content for policy violations and flag any issues.",
        "Check whether the text below breaches content guidelines.",
        "Assess this text against standard content moderation rules.",
        "Is there anything in this text that would be flagged by a moderator?",
        "Evaluate the following for content policy compliance.",
        "Would this text pass content review? Identify any concerns.",
        "Scan the text for potential content policy issues.",
        "Determine if the content below is safe for publication.",
        "Audit this text against content safety standards.",
    ]),
    ("moderation_pii", [
        "Identify and redact any personally identifiable information in the text.",
        "Find all PII in the passage and mask it.",
        "Scan the following for personal data and replace it with placeholders.",
        "Redact names, addresses, phone numbers, and other PII from this text.",
        "Remove any personally identifiable information from the text below.",
        "Strip out all PII — names, emails, SSNs, phone numbers, addresses.",
        "Locate personal information in the passage and anonymize it.",
        "Mask all sensitive personal data in the following text.",
        "Identify PII in the text and replace each instance with [REDACTED].",
        "Anonymize the text below by removing all identifiable personal details.",
    ]),

    # ---- planning / task decomposition (2 clusters) ------------------------
    ("planning_breakdown", [
        "Break this task into a step-by-step plan.",
        "Decompose the following goal into actionable subtasks.",
        "Create a detailed plan for accomplishing the task described.",
        "What steps would you take to complete this task? List them.",
        "Outline a step-by-step approach to the following objective.",
        "Turn this high-level goal into a concrete sequence of actions.",
        "Map out the steps needed to achieve the described outcome.",
        "Plan this out — what needs to happen first, second, third?",
        "Provide an ordered list of steps to accomplish this task.",
        "Develop an action plan for the following objective.",
    ]),
    ("planning_estimation", [
        "Estimate how long each step of this project will take.",
        "Provide time estimates for the tasks described below.",
        "How much effort would each phase of this project require?",
        "Give rough time estimates for completing each part of this plan.",
        "Estimate the duration of each step in the project breakdown.",
        "How long would you expect each of these tasks to take?",
        "Assign time estimates to each phase of the project.",
        "Provide effort estimates in hours or days for each subtask.",
        "Ballpark the time required for each step listed.",
        "Give a realistic time estimate for each component of this project.",
    ]),

    # ---- evaluation / scoring (2 clusters) ---------------------------------
    ("evaluation_rubric", [
        "Score the following response on a scale of 1-5 using the rubric provided.",
        "Evaluate this answer against the scoring criteria below.",
        "Rate the response using the specified rubric, with justification.",
        "Apply the rubric to assess the quality of this response.",
        "Grade the following answer on each dimension of the rubric.",
        "Use the criteria provided to score this response from 1 to 5.",
        "Assess the response quality according to the rubric and assign a score.",
        "Rate this answer using the evaluation framework described.",
        "Score the response on each rubric dimension and explain your rating.",
        "Evaluate and assign a numerical score based on the given criteria.",
    ]),
    ("evaluation_feedback", [
        "Provide constructive feedback on the following work.",
        "Review the submission below and give actionable suggestions for improvement.",
        "What are the strengths and weaknesses of this work? Give feedback.",
        "Critique the following and suggest specific improvements.",
        "Offer detailed feedback — what works, what doesn't, and what to change.",
        "Evaluate the quality of this work and provide improvement recommendations.",
        "Give honest, constructive criticism of the submission below.",
        "Review this work and identify areas for improvement with specific suggestions.",
        "Assess the submission and provide feedback that would help the author improve.",
        "What feedback would you give the author of this work?",
    ]),

    # ---- fact checking (1 cluster) -----------------------------------------
    ("fact_checking", [
        "Verify the factual claims in the following passage.",
        "Check whether the statements in this text are accurate.",
        "Fact-check the passage below and flag any inaccuracies.",
        "Are the claims made in this text correct? Verify each one.",
        "Identify any factual errors in the following passage.",
        "Review the text for accuracy and note any incorrect claims.",
        "Which statements in this passage are verifiably true, and which are not?",
        "Cross-check the facts presented in the text below.",
        "Assess the factual accuracy of each claim in the passage.",
        "Determine which claims in this text are supported by evidence and which are not.",
    ]),
]


# ---------------------------------------------------------------------------
# Expansion: structural transforms that genuinely change sentence shape
# ---------------------------------------------------------------------------

_STRUCTURAL_TRANSFORMS: list[tuple[str, callable]] = []


def _register_transform(name: str):
    def decorator(fn):
        _STRUCTURAL_TRANSFORMS.append((name, fn))
        return fn
    return decorator


@_register_transform("conditional_framing")
def _transform_conditional(text: str) -> str | None:
    """'If X, then Y' framing."""
    stripped = text.rstrip(".!?")
    if stripped.lower().startswith(("if ", "when ", "given ")):
        return None  # already conditional
    return f"Given the input below, {stripped[0].lower()}{stripped[1:]}."


@_register_transform("purpose_clause")
def _transform_purpose(text: str) -> str | None:
    """Add a purpose clause to reframe the instruction."""
    purposes = [
        "To help me understand the content",
        "So I can quickly review the key points",
        "For inclusion in a report",
        "To support downstream processing",
        "So the information is easier to work with",
    ]
    stripped = text.rstrip(".!?")
    purpose = random.choice(purposes)
    return f"{purpose}, {stripped[0].lower()}{stripped[1:]}."


@_register_transform("passive_voice")
def _transform_passive(text: str) -> str | None:
    """Attempt a passive-voice reframe."""
    lower = text.lower()
    # Only apply to imperative sentences starting with action verbs
    action_verbs = ["write", "create", "generate", "produce", "provide", "list",
                    "identify", "find", "extract", "determine", "classify", "translate"]
    starts_with_verb = any(lower.startswith(v) for v in action_verbs)
    if not starts_with_verb:
        return None
    stripped = text.rstrip(".!?")
    return f"The following task should be completed: {stripped[0].lower()}{stripped[1:]}."


@_register_transform("output_first")
def _transform_output_first(text: str) -> str | None:
    """Restructure so the output description comes first."""
    stripped = text.rstrip(".!?")
    outputs = [
        "The expected output is the result of the following",
        "What I need is for you to",
        "The deliverable here is to",
        "Your output should be the result of",
    ]
    prefix = random.choice(outputs)
    return f"{prefix}: {stripped[0].lower()}{stripped[1:]}."


@_register_transform("negation_framing")
def _transform_negation(text: str) -> str | None:
    """Frame via what NOT to do."""
    stripped = text.rstrip(".!?")
    negations = [
        f"Don't just skim the text — {stripped[0].lower()}{stripped[1:]}.",
        f"Rather than ignoring the details, {stripped[0].lower()}{stripped[1:]}.",
        f"Instead of paraphrasing loosely, {stripped[0].lower()}{stripped[1:]}.",
    ]
    return random.choice(negations)


def _apply_structural_transforms(
    base_variants: list[str],
    target_count: int,
    rng: random.Random,
) -> list[str]:
    """Expand a cluster by applying structural transforms to existing variants."""
    seen: set[str] = set()
    result: list[str] = []

    def add(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.strip())
        if normalized in seen:
            return False
        seen.add(normalized)
        result.append(text.strip())
        return True

    for v in base_variants:
        add(v)

    attempts = 0
    max_attempts = target_count * 10

    while len(result) < target_count and attempts < max_attempts:
        source = result[rng.randint(0, len(result) - 1)]
        _, transform_fn = _STRUCTURAL_TRANSFORMS[attempts % len(_STRUCTURAL_TRANSFORMS)]
        candidate = transform_fn(source)
        if candidate:
            add(candidate)
        attempts += 1

    return result[:target_count]


# ---------------------------------------------------------------------------
# Near-duplicate removal
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    """Jaccard similarity on whitespace tokens."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _deduplicate_cluster(
    variants: list[str],
    max_overlap: float = 0.92,
) -> list[str]:
    """Remove variants with very high token overlap within a cluster."""
    kept: list[str] = []
    for v in variants:
        if all(_token_overlap(v, k) < max_overlap for k in kept):
            kept.append(v)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_clusters(
    target_size: int = 500,
    n_clusters: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Build the evaluation dataset.

    Returns a list of records, each with:
        id, category, cluster_id, variant_id, text, is_canonical, canonical
    """
    rng = random.Random(seed)
    clusters = PARAPHRASE_CLUSTERS[:n_clusters]
    per_cluster = target_size // len(clusters)
    remainder = target_size % len(clusters)

    records: list[dict] = []

    for idx, (category, base_variants) in enumerate(clusters, start=1):
        target_variants = per_cluster + (1 if idx <= remainder else 0)

        # Expand via structural transforms if we need more than seed variants.
        expanded = _apply_structural_transforms(
            base_variants,
            max(target_variants, len(base_variants)),
            rng,
        )

        # Remove near-duplicates that would inflate unifier accuracy.
        deduped = _deduplicate_cluster(expanded)

        # If dedup removed too many, keep what we have.
        final = deduped[:target_variants]

        # Canonical = shortest variant (by whitespace token count).
        canonical = min(final, key=lambda t: len(t.split()))

        for vid, text in enumerate(final):
            records.append({
                "id": len(records),
                "category": category,
                "cluster_id": idx,
                "variant_id": vid,
                "text": text,
                "is_canonical": text == canonical,
                "canonical": canonical,
            })

    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic paraphrase dataset for PromptFuse evaluation."
    )
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_paraphrases.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--clusters", type=int, default=50)
    args = parser.parse_args()

    records = expand_clusters(
        target_size=args.target_size,
        n_clusters=args.clusters,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    n_clusters = len({r["cluster_id"] for r in records})
    avg_per = len(records) / n_clusters if n_clusters else 0
    print(f"Wrote {len(records)} paraphrases across {n_clusters} clusters to {args.output}")
    print(f"Average {avg_per:.1f} variants per cluster")

    # Quick diversity check.
    overlaps = []
    for cid in range(1, n_clusters + 1):
        cluster_texts = [r["text"] for r in records if r["cluster_id"] == cid]
        for i in range(len(cluster_texts)):
            for j in range(i + 1, len(cluster_texts)):
                overlaps.append(_token_overlap(cluster_texts[i], cluster_texts[j]))
    if overlaps:
        avg_overlap = sum(overlaps) / len(overlaps)
        print(f"Mean within-cluster token overlap: {avg_overlap:.3f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register
from datetime import datetime
now = datetime.now()
import json

@prompt_register.register_module()
class StoryWrtingNLWriterEN(PromptTemplate):
    def __init__(self) -> None:
        system_message = """
You are a professional and innovative writer collaborating with other writers to create a user-requested novel.  

### Requirements:
- Start from the previous ending of the story, matching the existing text's writing style, vocabulary, and overall atmosphere. Naturally complete your section according to the writing requirements, without reinterpreting or re-describing details or events already covered.
- Pay close attention to the existing novel design conclusions.
- Use rhetorical, linguistic, and literary devices (e.g., ambiguity, alliteration) to create engaging effects.
- Avoid plain or repetitive phrases (unless intentionally used to create narrative, thematic, or linguistic effects).
- Employ diverse and rich language: vary sentence structure, word choice, and vocabulary.
- Avoid summarizing, explanatory, or expository content or sentences unless absolutely necessary.
- Ensure there is no sense of disconnection or abruptness in the plot or descriptions. You may write some transitional content to maintain complete continuity with the existing material.
- The final output within the `<article>` tags must be clean, publishable-quality story text without any embedded meta-commentary or non-story elements.

### Instructions:
1.  First, conduct any necessary reflection or thinking within `<think></think>` tags. This section will be ignored for the final story output.
2.  Then, provide the story continuation **exclusively** within `<article></article>` tags.
3.  **Crucially, the content within the `<article></article>` tags MUST be only the pure, narrative story text, in the target language (e.g., Chinese if the story is in Chinese). It MUST NOT contain any of your thought processes, explanations, summaries, comments on the writing process, or any text that is not part of the story itself.** Such meta-text or explanatory content, if any, should be confined to the `<think></think>` section.
4.  The text within `<article></article>` will be directly used as the novel's content.
""".strip()

        
        content_template = """
The collaborative story-writing requirement to be completed:  
**{to_run_root_question}**  

Based on the existing novel design conclusions and the requirements, continue writing the story. You need to continue writing:  
**{to_run_task}**

---
The existing novel design conclusions are as follows, you should obey it:  
```
{to_run_outer_graph_dependent}

{to_run_same_graph_dependent}
```

---
already-written novel:
```
{to_run_article}
```

Based on the requirements, continue writing **{to_run_task}**.
"""
        super().__init__(system_message, content_template)
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState


class SystemPrompt:
	def __init__(
		self, action_description: str, current_date: datetime, max_actions_per_step: int = 10
	):
		self.default_action_description = action_description
		self.current_date = current_date
		self.max_actions_per_step = max_actions_per_step

	def important_rules(self) -> str:
		"""
		Returns the important rules for the agent.
		"""
		text = """
1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {
     "current_state": {
       "evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are succesful like intended by the task. Ignore the action result. The website is the ground truth. Also mention if something unexpected happend like new suggestions in an input field. Shortly state why/why not. Most importantly, all your actions may not be succesfully executed by the user, so despite the memory and input prompt, you need to re-evaluate the situation and decide whether you need to continue with the same actions or change them. Make sure to re-evaluate the situation and provide a reason for failure/unknown, for example: 'Problem:My search query is too complex due to the use of multiple logical operators and no results were found.' Then come up with a new solution to your problem, example: 'Solution: Try a simpler search query without quotation marks and logical operators to find more results and then filter the results.'",
       "memory": "Description of what has been done and what you need to remember until the end of the task",
       "next_goal": "What needs to be done with the next actions. First, reason about the current state and what element you would need to click and interact with using which action. Then, come up with the next goal. Then, think about how to achieve the next goal using actions."
     },
     "action": [
       {
         "one_action_name": {
           // action-specific parameter
         }
       },
       // ... more actions in sequence
     ]
   }

2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item. 

   Common action sequences:
   - Form filling: [
       {"input_text": {"index": 1, "text": "username"}},
       {"input_text": {"index": 2, "text": "password"}},
       {"click_element": {"index": 3}}
     ]
   - Navigation and extraction: [
       {"open_new_tab": {}},
       {"go_to_url": {"url": "https://example.com"}},
       {"extract_page_content": {}}
     ]


3. ELEMENT INTERACTION:
   - Only use indexes that exist in the provided element list
   - Each element has a unique index number (e.g., "33[:]<button>")
   - Elements marked with "_[:]" are non-interactive (for context only)

4. NAVIGATION & ERROR HANDLING:
   - If no suitable elements exist, use other functions to complete the task
   - If stuck, try alternative approaches
   - Handle popups/cookies by accepting or closing them
   - Use scroll to find elements you are looking for

5. TASK COMPLETION:
   - Use the done action as the last action as soon as the task is complete
   - Don't hallucinate actions
   - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
   - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.

6. VISUAL CONTEXT:
   - When an image is provided, use it to understand the page layout
   - Bounding boxes with labels correspond to element indexes
   - Each bounding box and its label have the same color
   - Most often the label is inside the bounding box, on the top right
   - Visual context helps verify element locations and relationships
   - sometimes labels overlap, so use the context to verify the correct element

7. Form filling:
   - If you fill an input field and your action sequence is interrupted, most often a list with suggestions poped up under the field and you need to first select the right element from the suggestion list.

8. Searching via input box:
	- If you fill out an input field for the purpose of searching, then you need to use send_keys with enter key to submit the search. Try using send_keys and only if that fails, use click_element to submit the search.

9. ACTION SEQUENCING:
   - Actions are executed in the order they appear in the list 
   - Each action should logically follow from the previous one
   - If the page changes after an action, the sequence is interrupted and you get the new state. This mean your previous actions may not be succesfully executed by the user. You must re-evaluate the situation and decide whether you need to continue with the same actions or change them.
   - If content only disappears the sequence continues.
   - Only provide the action sequence until you think the page will change. For example, if you type something in an input field, suggestions pop up and the page changes. You need to provide the action sequence until you think the page will change.
   - Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page like saving, extracting, checkboxes...
   - only use multiple actions if it makes sense. 

10. Troubleshooting: If you are stuck and an action fails, try to find the reason for the failure and try out a different approach. If you come up with a sequence of actions, and if your sequence is interrupted because of the page changing or something new appearing on the page, then propose an action sequence with only one single action.
"""
		text += f'   - use maximum {self.max_actions_per_step} actions per sequence'
		return text

	def input_format(self) -> str:
		return """
INPUT STRUCTURE:
1. Current URL: The webpage you're currently on
2. Available Tabs: List of open browser tabs
3. Interactive Elements: List in the format:
   index[:]<element_type>element_text</element_type>
   - index: Numeric identifier for interaction
   - element_type: HTML element type (button, input, etc.)
   - element_text: Visible text or element description

Example:
33[:]<button>Submit Form</button>
_[:] Non-interactive text


Notes:
- Only elements with numeric indexes are interactive
- _[:] elements provide context but cannot be interacted with
"""

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    str: Formatted system prompt
		"""
		time_str = self.current_date.strftime('%Y-%m-%d %H:%M')

		AGENT_PROMPT = f"""You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
1. Analyze the provided webpage elements and structure
2. Plan a sequence of actions to accomplish the given task
3. Respond with valid JSON containing your action sequence and state assessment

Current date and time: {time_str}

{self.input_format()}

{self.important_rules()}

Functions:
{self.default_action_description}

Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid."""
		return SystemMessage(content=AGENT_PROMPT)


# Example:
# {self.example_response()}
# Your AVAILABLE ACTIONS:
# {self.default_action_description}


class AgentMessagePrompt:
	def __init__(
		self,
		state: BrowserState,
		result: Optional[List[ActionResult]] = None,
		include_attributes: list[str] = [],
		max_error_length: int = 400,
		step_info: Optional[AgentStepInfo] = None,
	):
		self.state = state
		self.result = result
		self.max_error_length = max_error_length
		self.include_attributes = include_attributes
		self.step_info = step_info

	def get_user_message(self) -> HumanMessage:
		if self.step_info:
			step_info_description = (
				f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}'
			)
		else:
			step_info_description = ''

		elements_text = self.state.element_tree.clickable_elements_to_string(
			include_attributes=self.include_attributes
		)
		if elements_text != '':
			extra = '... Cut off - use extract content or scroll to get more ...'
			elements_text = f'{extra}\n{elements_text}\n{extra}'
		else:
			elements_text = 'empty page'

		state_description = f"""
{step_info_description}
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
Interactive elements from current page view:
{elements_text}
"""

		if self.result:
			for i, result in enumerate(self.result):
				if result.extracted_content:
					state_description += (
						f'\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}'
					)
				if result.error:
					# only use last 300 characters of error
					error = result.error[-self.max_error_length :]
					state_description += f'\nAction error {i + 1}/{len(self.result)}: ...{error}'

		if self.state.screenshot:
			# Format message for vision model
			return HumanMessage(
				content=[
					{'type': 'text', 'text': state_description},
					{
						'type': 'image_url',
						'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
					},
				]
			)
		

		return HumanMessage(content=state_description)
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor

import base64
import httpx
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib


class EmailSender:
    def __init__(self):
        pass

    def _format_addr(self, s: str):
        name, addr = parseaddr(s)
        return formataddr((Header(name, "utf-8").encode(), addr))

    def send_email(self, content: str):
        from_addr = "fanqingsong@gmail.com"
        password = "wkab culv iiza yjqt "
        to_addr = "qingsong.fan@tietoevry.com"
        smtp_server = "smtp.gmail.com"

        msg = MIMEText(content, "plain", "utf-8")
        msg["From"] = self._format_addr("FIRE ALARMER <%s>" % from_addr)
        msg["To"] = self._format_addr("ADMIN <%s>" % to_addr)
        msg["Subject"] = Header("You got one fire alarm!", "utf-8").encode()
        # print("---- before login 11-----")
        server = smtplib.SMTP_SSL(smtp_server, 465)
        # print("---- before login 1122-----")
        server.set_debuglevel(1)
        # print("---- before login -----")
        server.login(from_addr, password)
        # print("---- after login -----")
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()


class FireAlarmer:
    def __init__(self):
        # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.email_sender = EmailSender()

    def _get_tools(self):

        @tool
        def send_one_email(content: str) -> bool:
            """
            send one email to admin.

            Return the result of sending one email.
            """
            self.email_sender.send_email(content)

            return True

        tools = [send_one_email]

        return tools

    def _get_text_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are very powerful assistant, but don't know current events",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        return prompt

    def _get_image_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are very powerful assistant, but don't know current events",
                ),
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": "{input}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "{image_data}"},
                        }
                    ],
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        return prompt

    def _get_agent(self):
        tools = self._get_tools()
        prompt = self._get_image_prompt()
        llm = self.llm
        llm_with_tools = llm.bind_tools(tools)

        agent = (
            {
                "input": lambda x: x["input"],
                "image_data": lambda x: x['image_data'],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

        return agent

    def run(self, image_data: str):
        agent = self._get_agent()
        tools = self._get_tools()

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        user_prompt = '''
        please understand this image as of fire alarm, if any fire risk then send email with that fire description, otherwise do not send alarm email.
        finally tell me that fire situation in short beside email notice if any fire risk happened, otherwise just output "No Fire Alarm".
        '''

        ret = agent_executor.invoke({"input": user_prompt, "image_data": image_data})

        print(ret)

        return ret['output']


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    image_url = "https://media.licdn.com/dms/image/D4D12AQF0u0kA70nI_Q/article-cover_image-shrink_600_2000/0/1691063159568?e=2147483647&v=beta&t=KOWNla6Vhc_wRMTUil9X1NnPvzu3nt4eiv1DCmANNIU"
    # image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQhSUzTow_Xta3T3VF8op28XCTCM4D3boxhaA3ZrNyS5xkQXEpTNvsrUhmvfkwKOb5Z7jY&usqp=CAU"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    fire_alarmer = FireAlarmer()
    ret = fire_alarmer.run(image_data)
    print(ret)





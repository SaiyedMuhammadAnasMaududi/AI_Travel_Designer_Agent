from dataclasses import dataclass
import os
from pydantic import BaseModel
from typing import List,Any
import asyncio
from agents import Agent, Runner,function_tool,set_tracing_disabled,OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
set_tracing_disabled(True)
model="google/gemini-2.0-flash-001"
client=AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    )


@dataclass
class userdestination :
    """This class is used to store the user destination"""
    destination: str
    reason:str

@function_tool
async def get_flights(destination, date,):
    """ This function retreives flight schedule  according to te destination selected by the user also book the flight for it"""
    date = date.replace("-", "/")
    flight= f'Your flight for {destination} on {date} has been booked for You. Enjoy your trip!'
    return flight

@function_tool
async def get_hotels(destination, date,hotel_name):
    """ This function retreives hotel schedule  according to te destination selected by the user also book the hotel for it"""
    date = date.replace("-", "/")
    hotel= f'Your hotel : {hotel_name}  {destination} on {date} has been booked for You. Enjoy your stay!'
    return hotel 
 
hotel=Agent(
     name="suggest hotels",
     instructions="You suggest hotels using Websearch tools and return the hotels selected by th user",
     model=OpenAIChatCompletionsModel(
         openai_client=client,
         model=model,
         )
     

 )


destinationagent=Agent(
    name="suggest destination",
    instructions="You suggest destinations depending user mood and return the destination selected by the user",
    model=OpenAIChatCompletionsModel(
        openai_client=client,
        model=model,
        )

)

bookingagent=Agent(
    name="Booking Agents",
    instructions="according to the the destination selcted by the user, you bookk the flight for the user using get_flights tool and then suggest hotels to the user using thehote agent as tools and then taking the hotels selected by the user book that hotel using get_hotels tools," \
    "evrything should be done asynchronously   ",
    model=OpenAIChatCompletionsModel(
        openai_client=client,
        model=model,
        ),
    tools=[get_flights, get_hotels, hotel.as_tool(tool_name="hotel_agent",
                                                  tool_description="This tool is used to suggest hotels to the user based on the destination selected by the user "),],
)

ExploreAgents=Agent(
    name="Explore Agents",
    instructions="you are the agent that runs at the end and sugeests users the places and food to explore in the destination selected by the user.",
    model=OpenAIChatCompletionsModel(
        openai_client=client,
        model=model,
        )

)
@function_tool
async def ai_travel_planner(user_input: str):
    """This function is used to plan the travel for the user based on the user input"""
    destres=await Runner.run(destinationagent,input=user_input)
    destination=destres.final_output_as(userdestination).destination

    print(f"🌍 Suggested Destination: {destination}")

    # Step 2: Ask user to confirm or enter custom destination
    user_confirmed = input(f"❓ Do you want to proceed with '{destination}'? (yes / enter your own destination): ").strip().lower()
    
    if user_confirmed == "yes":
        selected_destination =destination
    else:
        selected_destination = user_confirmed 

    booking_result = await Runner.run(bookingagent, input_data=selected_destination)
    print(f"✅ Booking Done:\n{booking_result.final_output}")

    # Step 4: Explore
    explore_result = await Runner.run(ExploreAgents, input_data=selected_destination)
    print(f"✅ Explore Suggestions:\n{explore_result.final_output}")

    return f"🎯 Final Plan for {selected_destination}:\n\n📦 Bookings:\n{booking_result.final_output}\n\n🍽️ Explore:\n{explore_result.final_output}"     







aitaveldesigneragent =Agent(
    name="AI Travel Designer Agent",
    instructions="You are Multi-agents systems that allow user to work with multiple agents to design their travel plans. You will first suggest the destination to the user using destinationagent, then you will book the flight for the user using bookingagent and then you will suggest hotels to the user using hotel agent and then finally you will suggest places and food to explore in the destination selected by the user using ExploreAgents",
    model=OpenAIChatCompletionsModel(
        openai_client=client,
        model=model,),
        tools=[ai_travel_planner],
        handoffs=[bookingagent,ExploreAgents,destinationagent]
)


class Context(BaseModel):
    information:list[Any]=[]
    choices:List[Any]=[]




async def main():
    nput= input("How can I help You plan your trip ?")
    while True:
     result=await Runner.run(aitaveldesigneragent,input=nput,context=Context)
     print(result.final_output)
     nput=input("What other ?")

asyncio.run(main())    

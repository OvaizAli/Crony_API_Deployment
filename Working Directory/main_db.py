import asyncio
import motor.motor_asyncio

async def get_server_info():
# def get_database(colName):
    # from pymongo import MongoClient
    # import pymongo

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb+srv://OvaizAli:123@cronyai.idwl9.mongodb.net/test"

    # # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    # client = MongoClient(CONNECTION_STRING)

    client = motor.motor_asyncio.AsyncIOMotorClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    db = client.CronyAI
    col = db['action']
    cursor = col.find()

    try:
        data = await cursor.to_list(None)
        return [d for d in data]

    except Exception:
        return "Unable to connect to the server."

    # col = db[colName]
    # data = col.find()

    # for x in data:
    #     print(x)
    
# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":    
    
    # Get the database
    # get_database("action")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print(loop.run_until_complete(get_server_info()))
    

    # for x in data:
    #     print(x)
   
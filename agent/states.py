from pydantic import BaseModel, Field

class File(BaseModel):
    path: str = Field(description="path to file to be created or modified")
    purpose: str = Field(description="purpose of the file, e.g. 'main application logic', 'data processing module', etc.")

class Plan(BaseModel):
    name: str = Field(description="the name of the app to be built")
    description: str = Field(description="the description of the app to be built")
    techstack: str = Field(description="the techstack to be used for the app")
    features: list[str] = Field(
        description="the list of features to be used for the app, e.g. 'user authentication', 'data_visualization', etc.")
    files: list[File] = Field(description="a list of files to be created, each with a 'path' and 'purpose'")
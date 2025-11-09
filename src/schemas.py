from pydantic import BaseModel,field_validator


class Prompt(BaseModel):
    message:str

    @field_validator("message")
    @classmethod
    def validate_message(value,cls):
        if type(value)!=str:
            raise TypeError("Invalid data type ,expected string")
        
        return value

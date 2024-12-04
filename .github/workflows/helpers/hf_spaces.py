import os
import typer
from typer import Option, Argument
from typing import Annotated
from huggingface_hub import HfApi

cli = typer.Typer()
api = HfApi()

@cli.command()
def create_space(space_name: Annotated[str,Option()],
                 user_name: Annotated[str,Option()],
                 token: Annotated[str,Option()],
                 space_sdk: Annotated[str,Option()] = "gradio"):
    try:
        print(f"Creating space {space_name}")
        api.create_repo(repo_id=os.path.join(user_name,space_name),
                            repo_type="space",
                            space_sdk=space_sdk,
                            token=token)
        print(f"Space {space_name} created")
    except:
        print(f"An Error occured.")
        raise Exception("Error.")


@cli.command()
def upload_file(file_path: Annotated[str,Option()],
                path_in_repo: Annotated[str,Option()],
                token: Annotated[str,Option()],
                user_name: Annotated[str,Option()],
                space_name: Annotated[str,Option()]):
    try:
        print(f"Uploading file {file_path} to {path_in_repo}")
        api.upload_file(path_or_fileobj=file_path,
                          path_in_repo=path_in_repo,
                          repo_id=os.path.join(user_name,space_name),
                          repo_type="space",
                          token=token)
        print(f"File {file_path} uploaded to {path_in_repo}")
    except:
        print(f"File wasn't uploaded. An Error occured.")

@cli.command()
def upload_folder(folder_path: Annotated[str,Option()],
                path_in_repo: Annotated[str,Option()],
                token: Annotated[str,Option()],
                user_name: Annotated[str,Option()],
                space_name: Annotated[str,Option()]):
    try:
        print(f"Uploading directory {folder_path} to {path_in_repo}")
        api.upload_folder(folder_path=folder_path,
                          path_in_repo=path_in_repo,
                          repo_id=os.path.join(user_name,space_name),
                          repo_type="space",
                          token=token)
        print(f"Directory {folder_path} uploaded to {path_in_repo}")
    except:
        print(f"Directory wasn't uploaded. An Error occured.")
        raise Exception("Directory wasn't uploaded. An Error occured.")

# @cli.command()
# def create_space_dockerfile(from_file: Annotated[str,Option()],):
#     with open(from_file, "r") as f:
#         dockerfile_lines = f.readlines()
#     dockerfile_lines[-1] = 'ENTRYPOINT [ "python", "main.py", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]'

#     with open("Dockerfile", "w") as f:
#         f.writelines(dockerfile_lines)

if __name__ == "__main__":
    cli()
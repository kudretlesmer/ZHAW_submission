# Build Container

- docker build .
- docker compose build


# Run Container (-d for detach terminal)
- docker compose up -d


# Build & run
- docker compose up -d --build

# List running containers
- docker ps
- docker ps -a

# Enter container shell
- docker exec -it simulator bash       

# See docker logs
- docker logs <container_name>



# Activate pipenv env
- pipenv shell

# Remove container
- docker container rm <container_name>
# Get cluster nodes
- kubectl get node

# Get cluster pods
- kubectl get pods


# Push to container registry
- docker login
- docker tag module_1_server-simulator kudretesmer/simulator:latest
- docker push kudretesmer/simulator:latest


# Create resource 
- kubectl apply -f simulator_deployment.yml 

# k9s port forward shift+f


# Installl metric-server (need for hpa)
- kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml


# Create multiple resources at once
- kubectl apply -f k8s 
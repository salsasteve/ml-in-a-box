import click
import os
import subprocess

@click.command()
@click.option('--container-name', default='triton-server-23.08', help='Set the container name')
@click.option('--gpus', default='all', help='Set the GPUs to use')
@click.option('--ipc', default='host', help='Set the IPC option')
@click.option('--shm-size', default='2g', help='Set the shared memory size')
@click.option('--memlock', default='-1', help='Set the memory locking limit')
@click.option('--stack', default='67108864', help='Set the stack size limit')
@click.option('--host-ports', default='8000,8001,8002', help='Set the host ports (comma-separated)')
@click.option('--container-ports', default='8000,8001,8002', help='Set the container ports (comma-separated)')
@click.option('--volume', default=os.getcwd() + ':/apps', help='Set the volume to mount')
@click.option('--workdir', default='/apps', help='Set the working directory inside the container')
@click.option('--image', default='nvcr.io/nvidia/tritonserver:23.08-py3', help='Set the Docker image to use')
def run_triton_server(container_name, gpus, ipc, shm_size, memlock, stack, host_ports, container_ports, volume, workdir, image):
    """Script to run NVIDIA Triton Inference Server container with Docker."""

    # Convert comma-separated ports into a list
    host_ports = host_ports.split(',')
    container_ports = container_ports.split(',')

    # Construct port mapping string for Docker command
    ports_mapping = ' '.join([f'-p {hp}:{cp}' for hp, cp in zip(host_ports, container_ports)])

    # Construct the Docker run command
    docker_command = (
        f'docker run --gpus {gpus} '
        f'--name {container_name} '
        f'--rm '
        f'-it '
        f'--ipc={ipc} '
        f'--shm-size={shm_size} '
        f'--ulimit memlock={memlock} '
        f'--ulimit stack={stack} '
        f'{ports_mapping} '
        f'-v {volume} '
        f'-w {workdir} '
        f'{image}'
    )

    # Print the command (for debugging purposes)
    print(f"Running command: {docker_command}")

    # Run the Docker command
    try:
        subprocess.run(docker_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    run_triton_server()

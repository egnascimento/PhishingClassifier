# ################################################################
# Universidade Federal de Sao Carlos (UFSCAR)
# Aprendizado de Maquina - 2020
# Projeto Final

# Aluno: Eduardo Garcia do Nascimento
# RA/CPF: 22008732800
# ################################################################

# Arquivo com todas as funcoes e codigos diversos usados para suporte durante o trabalho.
import os
import time

def beep(times, freq):
    """
    Reproduz um som de frequência continua no PC

    Durante o trabalho foi muito comum chegar em algumas etapas que demoraram um ou mais minutos
    inclusive as buscas em grid, momentos esses ideias para pegar um café (ou dois).
    Pra tornar esse processo mais eficiente e ter algum tipo de alerta pra que eu retornasse à análise
    inseri essa função para avisar que a célula do notebook foi processada por completo.

    Usuário de Windows não terão essa conveniência a não ser que melhorem a função. :)
    """
    duration = 0.5  # seconds

    for _ in range(times):
        if os.name == 'posix': # Somente executa se não for Windows
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            time.sleep(0.5)

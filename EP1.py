"""
EP1 de numérico

Alunos: Murilo Costa Campos de Moura NUSP: 10705763
        Marina Botelho de Mesquita NUSP: 10771156

Professor: Clodoaldo Grotta Ragazzo
Turma: 08
"""
#importação dos pacotes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def decompoe_matriz(diag_A,subdiag_A):
    """Calcula a decomposição LDLt de uma matriz tridiagonal simétrica A

    Parameters:
    diag_A (array): diagonal principal da matriz A
    subdiag_A (array): subdiagonal da matriz A
    
    Returns:
    L(array): matriz L da decomposição armazenada em 1 vetor
    D(array): matriz D da decomposição armazenada em 1 vetor
   """
    D=np.empty(len(diag_A))
    L=np.empty(len(subdiag_A))
  
    for i in range(len(diag_A)):
        if i == 0:
            D[i]=diag_A[i]
            L[i]=subdiag_A[i]/D[i]
        elif i>0 and i < len(diag_A)-1:
            D[i]=diag_A[i]-L[i-1]**2*D[i-1]
            L[i]=subdiag_A[i]/D[i]
        else:
            D[i]=diag_A[i]-L[i-1]**2*D[i-1]
    
    return L,D

def resolve_sistema(L,D,b):
    """Calcula a solução de um sistema linear do tipo LDLt*x=b

    Parameters:
    L(array): matriz L da decomposição armazenada em 1 vetor
    D(array): matriz D da decomposição armazenada em 1 vetor
    b(array): matriz do lado direito do sistema
    
    Returns:
    x(array): vetor contendo a solução do sistema

   """
    z=np.empty(len(b))
    for i in range(len(z)):
        if i == 0:
            z[0]=b[0]
        else:
            z[i]=b[i]-L[i-1]*z[i-1]
    
    y=z/D
    x=np.empty(len(b))
    
    for i in range(len(b)):
        if i == 0:
            x[-1]=y[-1]
        else:
            x[-1-i]=y[-1-i]-L[-i]*x[-i]
    
    return x

def explicito(N,M,teste):
    """Fornece a solução aproximada usando o método explícito

    Parameters:
    N (int): número de passos em x
    M (int): número de passos em t
    teste (int): qual a equação que deverá ser resolvida (conforme está no LEIAME.txt)
        
    Returns:
    grafico(array): vetor com o valor da solução aproximada a cada 0.1s

   """
     
    x_vetor = np.linspace(0, 1, N+1)
    lamb=N**2/M
    delta_x=1/N
    delta_t=lamb/N**2
    t_vetor=np.linspace(0, 1, M+1)
    
    #localiza a posição da solução a cada 0.1s
    p=np.linspace(0,M,M+1)
    posicoes=np.zeros((1,M+1))
    for x in range (11):
         a=int(M/10*x)
         c=p==a
         posicoes=posicoes+c
    
    if teste == 1:
        #condição inicial u0
        grafico=x_vetor**2*(1-x_vetor)**2
        linha_ant=grafico 
        for i,t in enumerate(t_vetor):
            if t>0:
                x=linha_ant+lamb*(np.roll(linha_ant,1)-2*linha_ant+np.roll(linha_ant,-1))+delta_t*(10*np.cos(10*t)*x_vetor**2*(1-x_vetor)**2-(1+np.sin(10*t))*(12*x_vetor**2-12*x_vetor+2))
                #condição nas fronteiras
                x[0]=0
                x[-1]=0
                linha_ant=x
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
                
    elif teste == 2:
        #condição inicial u0
        grafico=np.exp(x_vetor*(-1))
        linha_ant=grafico 
        for i,t in enumerate(t_vetor):
            if t>0:
                x=linha_ant+lamb*(np.roll(linha_ant,1)-2*linha_ant+np.roll(linha_ant,-1))+delta_t*(np.exp(t-x_vetor)*(np.cos(5*t*x_vetor)*25*t**2-np.sin(5*t*x_vetor)*(10*t+5*x_vetor)))
                #condição nas fronteiras
                x[0]=np.exp(t)
                x[-1]=np.exp(t-1)*np.cos(5*t)
                linha_ant=x
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
                
    elif teste == 3:
        #condição inicial u0
        grafico=np.array([0]*(N+1))
        linha_ant=grafico
        #vetor booleano que localiza o ponto p
        p=np.logical_and(x_vetor >= 0.25-delta_x/2, x_vetor <= 0.25+delta_x/2)
        for i,t in enumerate(t_vetor):
            if t>0:
                x=linha_ant+lamb*(np.roll(linha_ant,1)-2*linha_ant+np.roll(linha_ant,-1))+delta_t/delta_x*p*10000*(1-2*t**2)
                #condição nas fronteiras
                x[0]=0
                x[-1]=0
                linha_ant=x
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
        
    return grafico

def euler(N,M,teste):
    """Fornece a solução aproximada usando o método de Euler implícito

    Parameters:
    N (int): número de passos em x
    M (int): número de passos em t
    teste (int): qual a equação que deverá ser resolvida (conforme está no LEIAME.txt)
        
    Returns:
    grafico(array): matriz (M+1)x(N+1) com o valor da solução aproximada a cada ponto

   """
    lamb=N**2/M
    delta_x=1/N
    delta_t=1/M
    t_vetor=np.linspace(0, 1, M+1)
    x_vetor=np.linspace(0, 1, N+1)
    intervalos=np.linspace(0, 1, 11)
    
    #localiza a posição da solução a cada 0.1s
    p=np.linspace(0,M,M+1)
    posicoes=np.zeros((1,M+1))
    for x in range (11):
         a=int(M/10*x)
         c=p==a
         posicoes=posicoes+c
    
    #definição da matriz do sistema linear
    diag_A=np.ones(N-1)*(1+2*lamb)
    subdiag_A=np.ones(N-2)*(-lamb)
    L,D=decompoe_matriz(diag_A,subdiag_A)
    
    #definições para a construção do gráfico
    x_vetormenor=x_vetor[1:-1]
    
    if teste == 1:
        #condição inicial u0
        grafico=(x_vetor**2*(1-x_vetor)**2)[1:-1]
        x_ant=grafico
        #condição nas fronteiras
        g=np.zeros((11,1))
        for i,t in enumerate(t_vetor):
            if t>0:
                b=delta_t*(10*np.cos(10*t)*x_vetormenor**2*(1-x_vetormenor)**2-(1+np.sin(10*t))*(12*x_vetormenor**2-12*x_vetormenor+2))+x_ant
                x=resolve_sistema(L,D,b)
                x_ant=x
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
        grafico=np.column_stack((g,grafico,g))
    
    elif teste == 2:
        #condição inicial u0
        grafico=np.exp(x_vetor*(-1))[1:-1]
        x_ant=grafico
        #condição nas fronteiras
        g1=np.exp(intervalos)
        g2=np.exp(intervalos-1)*np.cos(5*intervalos)
        for i,t in enumerate(t_vetor):
            if t>0:
                b=delta_t*(np.exp(t-x_vetormenor)*(np.cos(5*t*x_vetormenor)*25*t**2-np.sin(5*t*x_vetormenor)*(10*t+5*x_vetormenor)))+x_ant
                b[0]=b[0]+lamb*np.exp(t)
                b[-1]=b[-1]+lamb*np.exp(t-1)*np.cos(5*t)
                x=resolve_sistema(L,D,b)
                x_ant=x 
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
        grafico=np.column_stack((g1,grafico,g2))
    
    elif teste == 3:
        #condição inicial u0
        grafico=np.array([0]*(N-1))
        x_ant=grafico
        #condição nas fronteiras
        g=np.zeros((11,1))
        #vetor booleano que localiza o ponto p
        p=np.logical_and(x_vetor >= 0.25-delta_x/2, x_vetor <= 0.25+delta_x/2)[1:-1]  
        for i,t in enumerate(t_vetor):
            if t>0:
                b=delta_t/delta_x*p*10000*(1-2*t**2)+x_ant
                x=resolve_sistema(L,D,b)
                x_ant=x
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
        grafico=np.column_stack((g,grafico,g))
        
    return grafico
        
def crank(N,M,teste):
    """Fornece a solução aproximada usando o método de Crank-Nicolson

    Parameters:
    N (int): número de passos em x
    M (int): número de passos em t
    teste (int): qual a equação que deverá ser resolvida (conforme está no LEIAME.txt)
        
    Returns:
    grafico(array): matriz (M+1)x(N+1) com o valor da solução aproximada a cada ponto

   """
    lamb=N**2/M
    delta_x=1/N
    delta_t=lamb/N**2
    x_vetor = np.linspace(0, 1, N+1)  
    t_vetor=np.linspace(0, 1, int(1/delta_t)+1)
    intervalos=np.linspace(0, 1, 11)
    
    #localiza a posição da solução a cada 0.1s
    p=np.linspace(0,M,M+1)
    posicoes=np.zeros((1,M+1))
    for x in range (11):
         a=int(M/10*x)
         c=p==a
         posicoes=posicoes+c
    
    #definição da matriz do sistema linear
    diag_A=np.ones(N-1)*(1+lamb)
    subdiag_A=np.ones(N-2)*(-lamb/2)
    L,D=decompoe_matriz(diag_A,subdiag_A)
    
    #definições para a construção do gráfico
    x_vetormenor=x_vetor[1:-1]
    p_seminicio=x_vetormenor != x_vetormenor[0]
    p_semfim=x_vetormenor != x_vetormenor[-1]
    
    
    if teste == 1:
        #condição inicial u0
        grafico=(x_vetor**2*(1-x_vetor)**2)[1:-1]
        x_ant=grafico
        #condição nas fronteiras
        g=np.zeros((11,1))
        #valor da fonte f no instante 0
        f_ant=10*x_vetormenor**2*(1-x_vetormenor)**2-(12*x_vetormenor**2-12*x_vetormenor+2)
        
        for i,t in enumerate(t_vetor):
            if t>0:
                f=10*np.cos(10*t)*x_vetormenor**2*(1-x_vetormenor)**2-(1+np.sin(10*t))*(12*x_vetormenor**2-12*x_vetormenor+2)
                b=(1-lamb)*x_ant+lamb/2*(p_seminicio*np.roll(x_ant,1)+p_semfim*np.roll(x_ant,-1))+delta_t/2*(f+f_ant)
                x=resolve_sistema(L, D, b)
                x_ant=x
                f_ant=f
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
        grafico=np.column_stack((g,grafico,g))
    
    elif teste == 2:
        #condição inicial u0
        grafico=np.exp(x_vetor*(-1))[1:-1]
        x_ant=grafico
        #condição nas fronteiras
        g1=np.exp(t_vetor)
        g2=np.exp(t_vetor-1)*np.cos(5*t_vetor)
        #valor da fonte f no instante 0
        f_ant=np.array([0]*(N-1))
        p_inicio=x_vetormenor == x_vetormenor[0]
        p_fim=x_vetormenor == x_vetormenor[-1]
      
        for i,t in enumerate(t_vetor):
            if t>0:
                f=np.exp(t-x_vetormenor)*(np.cos(5*t*x_vetormenor)*25*t**2-np.sin(5*t*x_vetormenor)*(10*t+5*x_vetormenor))
                b=(1-lamb)*x_ant+lamb/2*(p_inicio*(g1[i]+g1[i-1])+p_seminicio*np.roll(x_ant,1)+p_semfim*np.roll(x_ant,-1)+p_fim*(g2[i]+g2[i-1]))+delta_t/2*(f+f_ant)
                x=resolve_sistema(L, D, b)
                x_ant=x
                f_ant=f
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
        g1_pequeno=np.exp(intervalos)
        g2_pequeno=np.exp(intervalos-1)*np.cos(5*intervalos)
        grafico=np.column_stack((g1_pequeno,grafico,g2_pequeno))
        
    elif teste == 3:
        #condição inicial u0
        grafico=np.array([0]*(N-1))
        x_ant=grafico
        #condição nas fronteiras
        g=np.zeros((11,1))
        #vetor booleano que localiza o ponto p
        p=np.logical_and(x_vetor >= 0.25-delta_x/2, x_vetor <= 0.25+delta_x/2)[1:-1]
        #valor da fonte f no instante 0
        f_ant=1/delta_x*p*10000
        
        for i,t in enumerate(t_vetor):
            if t>0:
                f=1/delta_x*p*10000*(1-2*t**2)
                b=(1-lamb)*x_ant+lamb/2*(p_seminicio*np.roll(x_ant,1)+p_semfim*np.roll(x_ant,-1))+delta_t/2*(f+f_ant)
                x=resolve_sistema(L, D, b)
                x_ant=x
                f_ant=f
                #solução a cada 0.1s
                if posicoes[0][i]==1:
                    grafico=np.vstack((grafico,x))
        grafico=np.column_stack((g,grafico,g))
            
    return grafico

def cria_grafico(grafico,N,M,metodo,teste,save):
    """Gera o gráfico da solução

    Parameters:
    grafico (array): matriz (M+1)x(N+1) com o valor da solução aproximada a cada ponto
    N (int): número de passos em x
    M (int): número de passos em t
    metodo (int): qual o método de solução usado
    teste (int): qual a equação que foi resolvida (conforme está no LEIAME.txt)
    save (str): se o gráfico deve ser salvo
    
    Returns:
    None

   """
    x_vetor = np.linspace(0, 1, N+1)
    #cria o grafico com as solucoes a cada 0.1s
    colors = plt.cm.nipy_spectral(np.linspace(0,1,11))
    a=[x/10 for x in range (0,11)]
    for i,vetor in enumerate(grafico):
        plt.plot(x_vetor, vetor,label='t='+str(i/10),color=colors[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1,prop={'size':11})    
    plt.xticks(np.linspace(0,1,11),a)
    plt.ylabel("Temperatura")
    plt.xlabel("x(m)")
    plt.title("Solução aproximada com λ="+str(int(N**2/M*100)/100)+" N="+ str(N))
    if save=='s':
        if metodo == 1 :
            plt.savefig("graf_metodo-"+str(metodo)+"_teste-"+str(teste)+"_λ-0,"+str(int(N**2/M*100))+"_N-"+ str(N), dpi=300,bbox_inches="tight")
        else:
            plt.savefig("graf_metodo-"+str(metodo)+"_teste-"+str(teste)+"_λ-"+str(M)+"_N-"+ str(N), dpi=300,bbox_inches="tight")
    plt.show()
    
   
def erro(grafico,teste,metodo,N,M):
    """Retorna o erro máximo no instante T=1s

    Parameters:
    grafico (array): matriz (M+1)x(N+1) com o valor da solução aproximada a cada ponto
    teste (int): qual a equação que foi resolvida (conforme está no LEIAME.txt)
    metodo (int): qual o método de solução usado
    N (int): número de passos em x
    M (int): número de passos em t

    Returns:
    e (float): valor máximo do erro

   """
    x_vetor = np.linspace(0, 1, N+1)
    if teste == 1:
        e=np.abs((1+np.sin(10))*x_vetor**2*(1-x_vetor)**2-grafico[-1])
    
    elif teste == 2:
        e=np.abs(np.exp(1-x_vetor)*np.cos(5*x_vetor)-grafico[-1])
     
    print("Erro:", max(e))
    return max(e)
 
def evol_erro(erro_vetor,teste,metodo,N,M,save):
    """Gera um gráfico com a evolução do erro em escala log e printa as estimativas do fator de redução do erro e ordem de convergência

    Parameters:
    erro_vetor (array): vetor com o valor máximo do erro para os diferentes N testados
    teste (int): qual a equação que foi resolvida (conforme está no LEIAME.txt)
    metodo (int): qual o método de solução usado
    N (int): número de passos em x
    M (int): número de passos em t
    save (str): se o gráfico deve ser salvo

    Returns:
    None

   """
    #plot da evolução do erro
    nums=[10*2**x for x in range(6)]
    fig1, ax1 = plt.subplots()
    ax1.plot(nums,erro_vetor,marker='.')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks(nums)
    ax1.set_yticks(erro_vetor)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel("Erro máximo")
    plt.xlabel("N")
    plt.title("Evolução do erro, metodo "+str(metodo)+" teste "+str(teste))
    #cálculo do fator de redução
    reducao=np.array([erro_vetor[x]/erro_vetor[x+1] for x in range(5)])
    print("Fator de redução do erro a cada refinamento da malha: ",reducao)
    #cálculo da estimativa da ordem de convergencia
    if metodo !=1:
        converg=np.array([np.log(erro_vetor[x+1]/erro_vetor[x])/np.log(nums[x]/nums[x+1]) for x in range(5)])
        print("Ordem de convergência: ",converg)
    if save=='s':
        if metodo == 1 :
            plt.savefig("evol_metodo-"+str(metodo)+"_teste-"+str(teste)+"_λ-0,"+str(int(N**2/M*100)), dpi=300,bbox_inches="tight")
        else:
            plt.savefig("evol_metodo-"+str(metodo)+"_teste-"+str(teste), dpi=300,bbox_inches="tight")
    
    plt.show()
    
def main():
    
    while True:
        
        erro_vetor=np.array([])
        metodo=int(input("Insira o número correspondente ao método que deseja utilizar (explícito = 1, Euler implícito = 2, Crank-Nicolson = 3): "))
        teste=int(input("Insira o número correspondente ao teste que deseja realizar (a = 1, b = 2, c = 3): "))
        save=input("Deseja salvar no computador os gráficos feitos ao longo dos testes? (s/n): ")
        
        #loop para testar as combinações de M e N 
        while True: 
            
            N=int(input("Insira o valor de N: "))
            lamb=float(input("Insira o valor de λ para determinar M: "))    
            M=int(N**2/lamb)
            
            if metodo == 1:
                grafico = explicito(N, M, teste)
                cria_grafico(grafico,N,M,metodo,teste,save)
                 
            elif metodo == 2:
                grafico = euler(N, M, teste)
                cria_grafico(grafico,N,M,metodo,teste,save)
                  
            elif metodo == 3:
                grafico = crank(N, M, teste)
                cria_grafico(grafico,N,M,metodo,teste,save)
            
            #salva os erros
            if teste != 3:
                erro_vetor=np.append(erro_vetor,erro(grafico,teste,metodo,N,M))
                #fim dos testes
                if N == 320 and len(erro_vetor)==6:
                    evol_erro(erro_vetor,teste,metodo,N,M,save)
            
            resp=input("Deseja testar com novos N e M? (s/n): ")   
            if resp == 'n':
                break
        
        resp=input("Deseja testar outro método de solução? (s/n): ")   
        if resp == 'n':
            break
main()   
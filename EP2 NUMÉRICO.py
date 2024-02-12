"""
EP2 de numérico

Alunos: Murilo Costa Campos de Moura NUSP: 10705763
        Marina Botelho de Mesquita NUSP: 10771156

Professor: Clodoaldo Grotta Ragazzo
Turma: 08
"""
#importação dos pacotes

import numpy as np
import matplotlib.pyplot as plt

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
    """Calcula a solução de um sistema linear do tipo LDLt*x=b,
    para uma matriz do sistema tridiagonal

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
   
def crank(N,p):
    """Fornece a solução aproximada usando o método de Crank-Nicolson

    Parameters:
    N (int): número de passos em x
    p (array): lista contendo os pontos p
        
    Returns:
    x_ant(array): matriz (N+1)x(nf) com o valor de uk para cada linha

   """
    M=N
    lamb=N**2/M
    delta_x=1/N
    delta_t=lamb/N**2
    x_vetor = np.linspace(0, 1, N+1)  
    t_vetor=np.linspace(0, 1, int(1/delta_t)+1)
        
    #definição da matriz do sistema linear
    diag_A=np.ones(N-1)*(1+lamb)
    subdiag_A=np.ones(N-2)*(-lamb/2)
    L,D=decompoe_matriz(diag_A,subdiag_A)
        
    #definições para a construção da solução
    x_vetormenor=x_vetor[1:-1]
    p_seminicio=x_vetormenor != x_vetormenor[0]
    p_semfim=x_vetormenor != x_vetormenor[-1]
    
    #criação de listas para guardar os valores relativos a cada ponto em p
    p_bool=[]
    f_ant=[]
    x_ant=[]
    for pk in p:
        posicao=np.logical_and(x_vetor >= pk-delta_x/2, x_vetor <= pk+delta_x/2)[1:-1]
        p_bool.append(posicao)
        f_ant.append(1/delta_x*posicao*20)
        x_ant.append(np.array([0]*(N-1)))
    
    for t in t_vetor:
        if t>0:
            for k in range(len(p)):
                f=10*(1+np.cos(5*t))/delta_x*p_bool[k]
                b=(1-lamb)*x_ant[k]+lamb/2*(p_seminicio*np.roll(x_ant[k],1)+p_semfim*np.roll(x_ant[k],-1))+delta_t/2*(f+f_ant[k])
                x=resolve_sistema(L, D, b)
                x_ant[k]=x
                f_ant[k]=f
                    
    return np.array(x_ant)

def matriz_MMQ(uk,uT):
    """Fornece a matriz e o lado direito do sistema de equações do MMQ

    Parameters:
    uk (array): matriz cujas colunas representam a solulção uk para cada um dos pontos fornecidos
    uT (array): array contendo a solução uT
        
    Returns:
    matriz(array): matriz do sistema linear
    rhs(array): lado direito do sistema linear
   """
    matriz=[]
    rhs=[]
    for i in range(len(uk)):
        linha=[]
        for j in range(len(uk)):
            #calculo dos produtos internos
            linha.append(np.dot(uk[i],uk[j]))
        #calculo dos produtos internos
        rhs.append(np.dot(uk[i],uT))
        matriz.append(linha)
        
    return np.array(matriz),np.array(rhs)

def decompoe_matriz_novo(matriz):
    """Calcula a decomposição LDLt de uma matriz simétrica qualquer

    Parameters:
    matriz (array): matriz simétrica que será decomposta
        
    Returns:
    L(array): matriz L da decomposição 
    D(array): matriz D da decomposição
   """
    L=np.diag(np.ones(len(matriz)))
    D=np.zeros(len(matriz))
    
    for i in range(len(matriz)):
        soma1=0
        for j in range (i):
            soma1+=L[i][j]**2*D[j]
        D[i]=matriz[i][i]-soma1
        for j in range (i+1,len(matriz)):
            soma2=0
            for k in range (i):
                soma2+=L[j][k]*D[k]*L[i][k]
            L[j][i]=(matriz[j][i]-soma2)/D[i]
    
    return L,D

def resolve_sistema_novo(L,D,b):
    """Calcula a solução de um sistema linear do tipo LDLt*x=b

    Parameters:
    L(array): matriz L da decomposição 
    D(array): matriz D da decomposição
    b(array): lado direito do sistema linear  
    
    Returns:
    x(array): array com o valor estimado para as intensidades das fontes
   """
    z=np.empty(len(b))
    
    for i in range(len(z)):
        soma=0
        for j in range (i):
            soma+=L[i][j]*z[j]
        z[i]=b[i]-soma
    
    y=z/D
    x=np.empty(len(b))
    for i in range(1,len(z)+1):
        soma=0
        for j in range (1,i):
            soma+=L[-j][-i]*x[-j]
        x[-i]=y[-i]-soma
    
    return x

def erro_quadratico(N,uT,uk,solucao):
    """Fornece o erro quadratico da solução obtida

    Parameters:
    N (int): número de passos em x
    uT (array): array contendo a solução uT
    uk (array): matriz cujas colunas representam a solulção uk para cada um dos pontos fornecidos
    solucao(array): array com o valor estimado para as intensidades das fontes
        
    Returns:
    E2(float): o erro quadrático

   """
    return np.sqrt(sum((uT-sum(np.matmul(np.diag(solucao),uk)))**2)*1/N)

def cria_grafico(N,uT,uk,solucao,save):
    """Gera os gráficos da solução aproximada e da solução reak

    Parameters:
    N (int): número de passos em x
    uT (array): array contendo a solução uT
    uk (array): matriz cujas colunas representam a solulção uk para cada um dos pontos fornecidos
    solucao(array): array com o valor estimado para as intensidades das fontes
    save (str): se o gráfico deve ser salvo
    
    Returns:
    None

   """
    x_vetor = np.linspace(0, 1, N+1)
    u_aprox=np.dot(solucao,uk)
    grafico=np.concatenate([[0],u_aprox,[0]])
    a=[x/10 for x in range (0,11)]
    
    #grafico da solução aproximada
    plt.plot(x_vetor, grafico) 
    plt.xticks(np.linspace(0,1,11),a)
    plt.ylabel("Temperatura")
    plt.xlabel("x")
    plt.title("Solução aproximada em T=1s e N="+ str(N))
    if save=='s':
            plt.savefig("graf_aprox_N-"+ str(N), dpi=300,bbox_inches="tight")
    plt.show()
    
    #grafico da solução real
    plt.plot(x_vetor, uT)
    plt.xticks(np.linspace(0,1,11),a)
    plt.ylabel("Temperatura")
    plt.xlabel("x")
    plt.title("Solução real em T=1s e N="+ str(N))
    if save=='s':
            plt.savefig("graf_real_N-"+ str(N), dpi=300,bbox_inches="tight")
    plt.show()
    
def main():
    teste=input("Qual caso deseja executar? (a, b, c ou d): ")
    
    if teste=="a":
        uk=crank(128,[0.35])
        uT=7*uk[0]
        matriz,b=matriz_MMQ(uk,uT)
        L,D=decompoe_matriz_novo(matriz)
        solucao=resolve_sistema_novo(L,D,b)
        print("Coeficiente:")
        print(solucao)
        erro=erro_quadratico(128,uT,uk,solucao)
    
    elif teste=="b":
        uk=crank(128,[0.15,0.3,0.7,0.8])
        uT=2.3*uk[0]+3.7*uk[1]+0.3*uk[2]+4.2*uk[3]
        matriz,b=matriz_MMQ(uk,uT)
        L,D=decompoe_matriz_novo(matriz)
        solucao=resolve_sistema_novo(L,D,b)
        print("Coeficientes:")
        print(solucao)
        erro=erro_quadratico(128,uT,uk,solucao)
    
    else:
        p=[]
        N=int(input("Qual N deseja testar? (128, 256, 512, 1024 ou 2048): "))
        save=input("Deseja salvar o grafico gerado? (s ou n): ") 
        #carrega o arquivo teste.txt 
        with open ("teste.txt","r") as texto:
            leitura=texto.read()
            lista=leitura.split()
            p=np.array(lista[:10]).astype(float)
            uT=np.array(lista[10:]).astype(float)
        
        if teste=="c":
            uTatual=[uT[i] for i in range(0, 2049, int(2048/N))]
            uTreduzido=uTatual[1:-1]
            uk=crank(N,p)
            matriz,b=matriz_MMQ(uk,uTreduzido)
            L,D=decompoe_matriz_novo(matriz)
            solucao=resolve_sistema_novo(L,D,b)
            print("Para N="+str(N)+":")
            print("Coeficientes:")
            for i in range (len(solucao)):
                print("a"+str(i)+"= " + str(solucao[i]))
            erro=erro_quadratico(N,uTreduzido,uk,solucao)
            print("Erro quadratico:",erro)
            print()
            cria_grafico(N,uTatual,uk,solucao,save)
        
        elif teste=="d":
            r=np.random.uniform(-1,1,2049)
            uT=uT*(1+0.01*r)
            uTatual=[uT[i] for i in range(0, 2049, int(2048/N))]
            uTreduzido=uTatual[1:-1]
            uk=crank(N,p)
            matriz,b=matriz_MMQ(uk,uTreduzido)
            L,D=decompoe_matriz_novo(matriz)
            solucao=resolve_sistema_novo(L,D,b)
            print("Para N="+str(N)+":")
            print("Coeficientes:")
            for i in range (len(solucao)):
                print("a"+str(i)+"= " + str(solucao[i]))
            erro=erro_quadratico(N,uTreduzido,uk,solucao)
            print("Erro quadratico:",erro)
            print()
            cria_grafico(N,uTatual,uk,solucao,save)
        
main()
            
    
            
        
        
      
        
    

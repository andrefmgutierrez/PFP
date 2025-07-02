# Long‑Term Investment Planner

Planejador de contribuições de longo prazo escrito em **Julia**.  
Recebe parâmetros de idade, renda e contribuição máxima e devolve
políticas ótimas comparadas a benchmarks tradicionais, gerando gráficos
de patrimônio e esforço de poupança.

## Pré‑requisitos

* Julia **≥ 1.9** (recomenda‑se a 1.11.x)
* Git para clonar o repositório

## Instalação rápida

```bash
git clone <REPO_URL>
cd long_term_planner_mvp
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Execução mínima

```bash
# JSON de exemplo (horizonte curtíssimo apenas para testar)
julia --project=. src/planner.jl examples/test_default.json
```

Os gráficos gerados ficam em `plots/`.

Para rodar com seus próprios parâmetros, copie `examples/test_default.json`
para `input.json`, edite os valores e execute:

```bash
julia --project=. src/planner.jl input.json
```

## Testes

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Estrutura essencial

```
├── src/                 # código‐fonte principal
├── data/                # séries históricas de mercado
├── examples/            # JSON(s) de entrada de exemplo
├── plots/               # saída de gráficos
└── test/                # suíte de testes unitários
```

## Licença

MIT – veja `LICENSE`.

## Autor

André Gutierrez (andregutierrez@…)

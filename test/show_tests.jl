@testitem "show: tree, constraints, colors, compact" tags=[:show] begin
    using AstroFit

    function EmissionLine(; center, flux=1.0)
        m = @model begin
            line = Gaussian1D(mean=center, amplitude=flux)
            line
        end
        @constrain m begin
            @bound line.sigma in (0, Inf)
        end
    end

    spectrum = @model begin
        cont = Linear1D(slope=0.0, intercept=1.0)
        Ha   = EmissionLine(center=6563.0)
        Hb   = EmissionLine(center=4861.0)
        cont + Ha + Hb
    end
    cm = @constrain spectrum begin
        @fix   cont.slope
        @bound Ha.line.sigma in (1, 5)
        @tie   Hb.line.sigma = Ha.line.sigma
    end

    out = sprint(show, MIME"text/plain"(), cm)
    @test occursin("CompiledModel  (6 free parameters)", out)
    @test occursin("cont :: Linear1D", out)
    @test occursin(r"slope\s+= 0\.0\s+fixed", out)
    @test occursin(r"intercept\s+= 1\.0\s+free", out)
    @test occursin("∈ [1, 5]", out)
    @test occursin("tied(Ha.line.sigma)", out)      # master named via registry
    @test occursin(r"Ha :: \w+\n\s+line :: Gaussian1D", out)  # prefab nesting

    # colors only when the IO asks for them
    col = sprint(show, MIME"text/plain"(), cm; context = :color => true)
    @test occursin("\e[32mfree\e[39m", col)
    @test occursin("\e[90mfixed\e[39m", col)
    @test occursin("\e[33m∈ [1, 5]\e[39m", col)
    @test occursin("\e[35mtied(Ha.line.sigma)\e[39m", col)
    @test !occursin("\e[", out)

    # compact form and ComponentRef
    @test sprint(show, cm) == "CompiledModel(3 components, 6 free)"
    refout = sprint(show, MIME"text/plain"(), cm.Ha.line)
    @test occursin("ComponentRef :: Gaussian1D", refout)
    @test occursin(r"mean\s+= 6563\.0\s+free", refout)

    # naked model: implicit flat registry, constraints shown
    ng = @constrain Gaussian1D() begin
        @fix   sigma
        @bound amplitude in (0, 10)
    end
    nout = sprint(show, MIME"text/plain"(), ng)
    @test occursin(r"amplitude\s+= 1\.0\s+∈ \[0, 10\]", nout)
    @test occursin(r"sigma\s+= 1\.0\s+fixed", nout)
end

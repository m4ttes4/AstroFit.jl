using Random

function gaussian1_lib!(out, xs)
    p = zeros(3)
    for _ in 1:100
        randn!(p)
        render!(out, withparams(CM_WP_1, p), xs)
    end
    out
end

function gaussian1_hw!(out, xs)
    p = zeros(3)
    for _ in 1:100
        randn!(p)
        A, μ, σ = p[1], p[2], p[3]
        @inbounds for i in eachindex(out, xs)
            out[i] = A * exp(-((xs[i] - μ) / σ)^2 / 2)
        end
    end
    out
end

function gaussian2_lib!(out, xs)
    p = zeros(6)
    for _ in 1:100
        randn!(p)
        render!(out, withparams(CM_WP_2, p), xs)
    end
    out
end

function gaussian2_hw!(out, xs)
    p = zeros(6)
    for _ in 1:100
        randn!(p)
        A1,μ1,σ1 = p[1],p[2],p[3]
        A2,μ2,σ2 = p[4],p[5],p[6]
        @inbounds for i in eachindex(out, xs)
            x = xs[i]
            out[i] = A1*exp(-((x-μ1)/σ1)^2/2) + A2*exp(-((x-μ2)/σ2)^2/2)
        end
    end
    out
end

function gaussian4_lib!(out, xs)
    p = zeros(12)
    for _ in 1:100
        randn!(p)
        render!(out, withparams(CM_WP_4, p), xs)
    end
    out
end

function gaussian4_hw!(out, xs)
    p = zeros(12)
    for _ in 1:100
        randn!(p)
        A1,μ1,σ1 = p[1],p[2],p[3]
        A2,μ2,σ2 = p[4],p[5],p[6]
        A3,μ3,σ3 = p[7],p[8],p[9]
        A4,μ4,σ4 = p[10],p[11],p[12]
        @inbounds for i in eachindex(out, xs)
            x = xs[i]
            out[i] = A1*exp(-((x-μ1)/σ1)^2/2) + A2*exp(-((x-μ2)/σ2)^2/2) +
                     A3*exp(-((x-μ3)/σ3)^2/2) + A4*exp(-((x-μ4)/σ4)^2/2)
        end
    end
    out
end

function gaussian64_lib!(out, xs)
    p = zeros(192)
    for _ in 1:100
        randn!(p)
        render!(out, withparams(CM_WP_64, p), xs)
    end
    out
end

function gaussian64_hw!(out, xs)
    p = zeros(192)
    for _ in 1:100
        randn!(p)
        @inbounds for i in eachindex(out, xs)
            x = xs[i]
            acc = 0.0
            for k in 1:64
                A, μ, σ = p[3k-2], p[3k-1], p[3k]
                acc += A * exp(-((x - μ) / σ)^2 / 2)
            end
            out[i] = acc
        end
    end
    out
end

SUITE["grid"] = BenchmarkGroup()

let o128=similar(XS_128), o1024=similar(XS_1024), o8192=similar(XS_8192)
    SUITE["grid"]["1G/128pts/lib"]  = @benchmarkable gaussian1_lib!($o128,  $XS_128)
    SUITE["grid"]["1G/128pts/hw"]  = @benchmarkable gaussian1_hw!($o128,  $XS_128)
    SUITE["grid"]["1G/1024pts/lib"] = @benchmarkable gaussian1_lib!($o1024, $XS_1024)
    SUITE["grid"]["1G/1024pts/hw"] = @benchmarkable gaussian1_hw!($o1024, $XS_1024)
    SUITE["grid"]["1G/8192pts/lib"] = @benchmarkable gaussian1_lib!($o8192, $XS_8192)
    SUITE["grid"]["1G/8192pts/hw"] = @benchmarkable gaussian1_hw!($o8192, $XS_8192)

    SUITE["grid"]["2G/128pts/lib"]  = @benchmarkable gaussian2_lib!($o128,  $XS_128)
    SUITE["grid"]["2G/128pts/hw"]  = @benchmarkable gaussian2_hw!($o128,  $XS_128)
    SUITE["grid"]["2G/1024pts/lib"] = @benchmarkable gaussian2_lib!($o1024, $XS_1024)
    SUITE["grid"]["2G/1024pts/hw"] = @benchmarkable gaussian2_hw!($o1024, $XS_1024)
    SUITE["grid"]["2G/8192pts/lib"] = @benchmarkable gaussian2_lib!($o8192, $XS_8192)
    SUITE["grid"]["2G/8192pts/hw"] = @benchmarkable gaussian2_hw!($o8192, $XS_8192)

    SUITE["grid"]["4G/128pts/lib"]  = @benchmarkable gaussian4_lib!($o128,  $XS_128)
    SUITE["grid"]["4G/128pts/hw"]  = @benchmarkable gaussian4_hw!($o128,  $XS_128)
    SUITE["grid"]["4G/1024pts/lib"] = @benchmarkable gaussian4_lib!($o1024, $XS_1024)
    SUITE["grid"]["4G/1024pts/hw"] = @benchmarkable gaussian4_hw!($o1024, $XS_1024)
    SUITE["grid"]["4G/8192pts/lib"] = @benchmarkable gaussian4_lib!($o8192, $XS_8192)
    SUITE["grid"]["4G/8192pts/hw"] = @benchmarkable gaussian4_hw!($o8192, $XS_8192)

    SUITE["grid"]["64G/128pts/lib"]  = @benchmarkable gaussian64_lib!($o128,  $XS_128)
    SUITE["grid"]["64G/128pts/hw"]  = @benchmarkable gaussian64_hw!($o128,  $XS_128)
    SUITE["grid"]["64G/1024pts/lib"] = @benchmarkable gaussian64_lib!($o1024, $XS_1024)
    SUITE["grid"]["64G/1024pts/hw"] = @benchmarkable gaussian64_hw!($o1024, $XS_1024)
    SUITE["grid"]["64G/8192pts/lib"] = @benchmarkable gaussian64_lib!($o8192, $XS_8192)
    SUITE["grid"]["64G/8192pts/hw"] = @benchmarkable gaussian64_hw!($o8192, $XS_8192)
end

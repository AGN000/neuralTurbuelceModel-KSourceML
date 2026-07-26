#include "kOmegaSSTML.H"
#include "fvMatrices.H"
#include "fvm.H"

namespace Foam
{
namespace RASModels
{

template<class BasicTurbulenceModel>
kOmegaSSTML<BasicTurbulenceModel>::kOmegaSSTML
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    kOmegaSST<BasicTurbulenceModel>
    (
        alpha, rho, U, alphaRhoPhi, phi, transport, propertiesName, type
    )
{
    if (type == typeName)
    {
        this->printCoeffs(type);
        Info << "kOmegaSSTML: ML source injection active." << nl
             << "  k    source field : kSourceML     [0 2 -3 0 0 0 0]" << nl
             << "  omega source field: omegaSourceML [0 0 -2 0 0 0 0]" << nl;
    }
}

template<class BasicTurbulenceModel>
tmp<volScalarField>
kOmegaSSTML<BasicTurbulenceModel>::tryReadSource
(
    const word& name,
    const dimensionSet& dims
) const
{
    const fvMesh& mesh = this->mesh_;
    const Time&   time = this->runTime_;

    IOobject io
    (
        name,
        time.name(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    );

    if (io.typeHeaderOk<volScalarField>(true))
    {
        return tmp<volScalarField>(new volScalarField(io, mesh));
    }

    IOobject io0
    (
        name,
        "0",
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE
    );

    if (io0.typeHeaderOk<volScalarField>(true))
    {
        return tmp<volScalarField>(new volScalarField(io0, mesh));
    }

    return tmp<volScalarField>(nullptr);
}

template<class BasicTurbulenceModel>
tmp<fvScalarMatrix>
kOmegaSSTML<BasicTurbulenceModel>::kSource() const
{
    tmp<fvScalarMatrix> tResult =
        kOmegaSST<BasicTurbulenceModel>::kSource();

    tmp<volScalarField> tSrc = tryReadSource
    (
        "kSourceML",
        dimArea / pow3(dimTime)
    );

    if (tSrc.valid())
    {
        const volScalarField& Src = tSrc();
        tResult.ref() += fvm::Su(this->alpha_() * this->rho_() * Src(), this->k_);
        Info << "kOmegaSSTML: kSourceML  mean=" << gAverage(Src)
             << "  max=" << gMax(Src) << nl;
    }

    return tResult;
}

template<class BasicTurbulenceModel>
tmp<fvScalarMatrix>
kOmegaSSTML<BasicTurbulenceModel>::omegaSource() const
{
    tmp<fvScalarMatrix> tResult =
        kOmegaSST<BasicTurbulenceModel>::omegaSource();

    tmp<volScalarField> tSrc = tryReadSource
    (
        "omegaSourceML",
        dimless / sqr(dimTime)
    );

    if (tSrc.valid())
    {
        const volScalarField& Src = tSrc();

        tResult.ref() +=
            fvm::Su
            (
                this->alpha_() * this->rho_()
                * max(Src, dimensionedScalar("zero", Src.dimensions(), Zero)),
                this->omega_
            );

        const dimensionedScalar omegaSmall
        (
            "omegaSmall", this->omega_.dimensions(), scalar(1e-10)
        );
        tResult.ref() +=
            fvm::Sp
            (
                this->alpha_() * this->rho_()
                * max(-Src, dimensionedScalar("zero", Src.dimensions(), Zero))
                / max(this->omega_, omegaSmall),
                this->omega_
            );

        Info << "kOmegaSSTML: omegaSourceML  mean=" << gAverage(Src)
             << "  max=" << gMax(Src) << nl;
    }

    return tResult;
}

} // End namespace RASModels
} // End namespace Foam

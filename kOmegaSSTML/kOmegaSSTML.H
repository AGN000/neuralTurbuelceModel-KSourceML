#ifndef kOmegaSSTML_H
#define kOmegaSSTML_H

#include "kOmegaSST.H"

namespace Foam
{
namespace RASModels
{

template<class BasicTurbulenceModel>
class kOmegaSSTML
:
    public kOmegaSST<BasicTurbulenceModel>
{
public:
    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField   rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    TypeName("kOmegaSSTML");

    kOmegaSSTML
    (
        const alphaField& alpha,
        const rhoField& rho,
        const volVectorField& U,
        const surfaceScalarField& alphaRhoPhi,
        const surfaceScalarField& phi,
        const transportModel& transport,
        const word& propertiesName = turbulenceModel::propertiesName,
        const word& type = typeName
    );

    virtual ~kOmegaSSTML() {}

protected:
    virtual tmp<fvScalarMatrix> kSource()     const override;
    virtual tmp<fvScalarMatrix> omegaSource() const override;

private:
    tmp<volScalarField> tryReadSource
    (
        const word& name,
        const dimensionSet& dims
    ) const;
};

} // End namespace RASModels
} // End namespace Foam

#ifdef NoRepository
    #include "kOmegaSSTML.C"
#endif

#endif

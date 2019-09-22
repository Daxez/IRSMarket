"""File containing all the models we need for the database"""
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, func, Float, Boolean, create_engine,BigInteger
from sqlalchemy.orm import relationship, backref, sessionmaker,scoped_session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RunModel(Base):
    __tablename__= "run"
    run_id = Column(String(36),primary_key=True)
    aggregate_id = Column(String(36))
    time_stamp = Column(DateTime)
    no_steps = Column(Integer)
    no_banks = Column(Integer)
    sigma = Column(Float)
    irs_threshold = Column(Float)
    max_irs_value = Column(Float)
    max_tenure = Column(Integer)
    threshold = Column(Float)
    seed = Column(BigInteger)
    market_type = Column(Integer)
    average_swaps = Column(Float)
    swaps = Column(BigInteger)
    dissipation = Column(Float)

    def __init__(self,rid,aggregate_id,no_steps,no_banks,sigma,irs_threshold,
                max_irs_value,max_tenure,threshold,time_stamp,seed,market_type,
                average_swaps,swaps,dissipation):
        self.run_id = rid
        self.aggregate_id = aggregate_id
        self.no_steps = no_steps
        self.no_banks = no_banks
        self.sigma = sigma
        self.irs_threshold = irs_threshold
        self.max_irs_value = max_irs_value
        self.max_tenure = max_tenure
        self.threshold = threshold
        self.time_stamp = time_stamp
        self.seed = seed
        self.market_type = market_type
        self.average_swaps = average_swaps
        self.swaps = swaps
        self.dissipation = dissipation

class BankModel(Base):
    __tablename__= "bank"
    bank_id = Column(String(36),primary_key=True)
    run_id = Column(String(36),primary_key=True)
    seed = Column(BigInteger)

    def __init__(self,bank_id,run_id,seed):
        self.bank_id = bank_id
        self.run_id = run_id
        self.seed = seed

class SwapModel(Base):
    __tablename__= "swap"
    swap_id = Column(String(36),primary_key=True)
    run_id = Column(String(36),primary_key=True)
    value = Column(Float)
    float_bank = Column(String(36))
    fix_bank = Column(String(36))
    start_time = Column(Integer)
    end_time = Column(Integer)
    tenure = Column(Integer)

    def __init__(self,swap_id,value,float_bank,fix_bank,start_time,end_time,tenure,run_id):
        self.swap_id = swap_id
        self.value = value
        self.float_bank = float_bank
        self.fix_bank = fix_bank
        self.start_time = start_time
        self.end_time = end_time
        self.tenure = tenure
        self.run_id = run_id

class DefaultModel(Base):
    __tablename__= "defaults"
    default_id = Column(String(36),primary_key=True)
    run_id = Column(String(36),primary_key=True)
    size = Column(Integer)
    time = Column(Integer)
    depth = Column(Integer, default=0)

    def __init__(self,default_id,size,time,run_id, depth):
        self.default_id = default_id
        self.size = size
        self.time = time
        self.run_id = run_id
        self.depth = depth

class BankDefaultModel(Base):
    __tablename__= "bank_default"
    default_id = Column(String(36),primary_key=True)
    bank_id = Column(String(36),primary_key=True)
    run_id = Column(String(36),primary_key=True)
    balance = Column(Float)
    root = Column(Boolean,default=False)

    def __init__(self,default_id,bank_id,balance,root,run_id):
        self.default_id = default_id
        self.bank_id = bank_id
        self.balance = balance
        self.root = root
        self.run_id = run_id

class DefaultAggregateModel(Base):
    __tablename__= "default_aggregate"
    aggregate_id = Column(String(36),primary_key=True)
    size = Column(Integer)
    frequency = Column(Integer)

    def __init__(self,aggregate_id,size,frequency):
        self.aggregate_id = aggregate_id
        self.size = size
        self.frequency = frequency

class AggregatePowerlaw(Base):
    __tablename__ = "aggregate_powerlaw"

    aggregate_id = Column(String(36),primary_key=True)
    powerlaw_index = Column(Integer)
    alpha = Column(Float)
    error = Column(Float)

    def __init__(self,aggregate_id,powerlaw_index,alpha,error):
        self.aggregate_id = aggregate_id

        if(powerlaw_index == None):
            powerlaw_index = -1
        self.powerlaw_index = powerlaw_index
        self.alpha = alpha
        self.error = error

class AggregateType(Base):
    __tablename__ = "aggregate_type"

    aggregate_id = Column(String(36),primary_key=True)
    type_id = Column(Integer)

    def __init__(self,aggregate_id,type_id):
        self.aggregate_id = aggregate_id
        self.type_id = type_id

class AggregateHumpWeight(Base):
    __tablename__ = "aggregateHumpWeight"

    aggregate_id = Column(String(36),primary_key=True)
    weight = Column(Float)
    alpha = Column(Float)

    def __init__(self,aggregate_id,weight, alpha):
        self.aggregate_id = aggregate_id
        self.weight = weight
        self.alpha = alpha

class AggregateDistribution(Base):
    __tablename__ = "aggregate_distribution"

    aggregate_id = Column(String(36),primary_key=True)
    w = Column(Float)
    alpha = Column(Float)
    mu = Column(Float)
    sigma = Column(Float)
    x0 = Column(Float, nullable=True)
    m = Column(Float, nullable=True)
    s = Column(Float, nullable=True)
    wsum = Column(Float, nullable=True)

    def __init__(self,aggregate_id,w,alpha,mu,sigma,x0,m,s,wsum):
        self.aggregate_id =aggregate_id
        self.w = w
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.m = m
        self.s = s
        self.wsum = wsum

conn = 'mysql://localhost:3306/IRS'
engine = create_engine(conn,pool_recycle=60)
Session = scoped_session(sessionmaker(expire_on_commit=False))
Session.configure(bind=engine)

def get_session():
    return Session()

if __name__ == '__main__':
    engine = create_engine('mysql://localhost:3306/IRS')

    session = sessionmaker(expire_on_commit=False)
    session.configure(bind=engine)
    Base.metadata.create_all(engine)
